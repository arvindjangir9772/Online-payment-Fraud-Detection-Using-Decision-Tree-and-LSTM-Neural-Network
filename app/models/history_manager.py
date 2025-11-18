"""
Transaction History Manager
Tracks and stores transaction check history for analytics and auditing
"""
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class HistoryManager:
    def __init__(self, db_path: str = "transaction_history.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for transaction history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create transactions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    transaction_data TEXT NOT NULL,
                    prediction INTEGER NOT NULL,
                    probability REAL NOT NULL,
                    lstm_probability REAL,
                    dt_probability REAL,
                    alpha_used REAL,
                    threshold_used REAL,
                    dominant_model TEXT,
                    explanation TEXT,
                    input_method TEXT,
                    session_id TEXT
                )
            ''')
            
            # Create analytics table for aggregated data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    total_transactions INTEGER DEFAULT 0,
                    fraud_detected INTEGER DEFAULT 0,
                    avg_probability REAL DEFAULT 0.0,
                    lstm_dominant_count INTEGER DEFAULT 0,
                    dt_dominant_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("History database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize history database: {e}")
    
    def add_transaction(self, 
                       transaction_data: Dict[str, Any],
                       prediction_result: Dict[str, Any],
                       input_method: str = "manual",
                       session_id: str = None) -> int:
        """Add a new transaction to history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Generate session ID if not provided
            if not session_id:
                session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            cursor.execute('''
                INSERT INTO transactions (
                    timestamp, transaction_data, prediction, probability,
                    lstm_probability, dt_probability, alpha_used, threshold_used,
                    dominant_model, explanation, input_method, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                json.dumps(transaction_data),
                prediction_result.get('prediction', 0),
                prediction_result.get('probability', 0.0),
                prediction_result.get('lstm_probability', 0.0),
                prediction_result.get('dt_probability', 0.0),
                prediction_result.get('alpha_used', 0.6),
                prediction_result.get('threshold_used', 0.5),
                prediction_result.get('dominant_model', 'unknown'),
                prediction_result.get('explanation', ''),
                input_method,
                session_id
            ))
            
            transaction_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.info(f"Transaction {transaction_id} added to history")
            return transaction_id
            
        except Exception as e:
            logger.error(f"Failed to add transaction to history: {e}")
            return -1
    
    def get_recent_transactions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent transaction history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM transactions 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            transactions = []
            for row in rows:
                transaction = dict(zip(columns, row))
                # Parse JSON data
                transaction['transaction_data'] = json.loads(transaction['transaction_data'])
                transactions.append(transaction)
            
            conn.close()
            return transactions
            
        except Exception as e:
            logger.error(f"Failed to get recent transactions: {e}")
            return []
    
    def get_daily_analytics(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get daily analytics for the past N days"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as total_transactions,
                    SUM(prediction) as fraud_detected,
                    AVG(probability) as avg_probability,
                    SUM(CASE WHEN dominant_model = 'lstm' THEN 1 ELSE 0 END) as lstm_dominant_count,
                    SUM(CASE WHEN dominant_model = 'dt' THEN 1 ELSE 0 END) as dt_dominant_count
                FROM transactions 
                WHERE timestamp >= datetime('now', '-{} days')
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            '''.format(days))
            
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            analytics = []
            for row in rows:
                analytics.append(dict(zip(columns, row)))
            
            conn.close()
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get daily analytics: {e}")
            return []
    
    def get_fraud_patterns(self) -> Dict[str, Any]:
        """Get fraud pattern analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get fraud rate by hour
            cursor.execute('''
                SELECT 
                    strftime('%H', timestamp) as hour,
                    COUNT(*) as total,
                    SUM(prediction) as fraud_count
                FROM transactions 
                GROUP BY strftime('%H', timestamp)
                ORDER BY hour
            ''')
            
            fraud_by_hour = []
            for row in cursor.fetchall():
                hour, total, fraud_count = row
                fraud_rate = (fraud_count / total) * 100 if total > 0 else 0
                fraud_by_hour.append({
                    'hour': int(hour),
                    'total_transactions': total,
                    'fraud_count': fraud_count,
                    'fraud_rate': round(fraud_rate, 2)
                })
            
            # Get fraud rate by amount ranges
            cursor.execute('''
                SELECT 
                    CASE 
                        WHEN json_extract(transaction_data, '$.Amount') < 50 THEN '0-50'
                        WHEN json_extract(transaction_data, '$.Amount') < 200 THEN '51-200'
                        WHEN json_extract(transaction_data, '$.Amount') < 1000 THEN '201-1000'
                        ELSE '1000+'
                    END as amount_range,
                    COUNT(*) as total,
                    SUM(prediction) as fraud_count
                FROM transactions 
                GROUP BY amount_range
                ORDER BY amount_range
            ''')
            
            fraud_by_amount = []
            for row in cursor.fetchall():
                amount_range, total, fraud_count = row
                fraud_rate = (fraud_count / total) * 100 if total > 0 else 0
                fraud_by_amount.append({
                    'range': amount_range,
                    'total_transactions': total,
                    'fraud_count': fraud_count,
                    'fraud_rate': round(fraud_rate, 2)
                })
            
            conn.close()
            
            return {
                'fraud_by_hour': fraud_by_hour,
                'fraud_by_amount_range': fraud_by_amount
            }
            
        except Exception as e:
            logger.error(f"Failed to get fraud patterns: {e}")
            return {'fraud_by_hour': [], 'fraud_by_amount_range': []}
    
    def get_total_stats(self) -> Dict[str, Any]:
        """Get overall statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_transactions,
                    SUM(prediction) as fraud_detected,
                    AVG(probability) as avg_probability,
                    MIN(timestamp) as first_transaction,
                    MAX(timestamp) as last_transaction
                FROM transactions
            ''')
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                total, fraud, avg_prob, first, last = row
                fraud_rate = (fraud / total) * 100 if total > 0 else 0
                
                return {
                    'total_transactions': total,
                    'fraud_detected': fraud,
                    'fraud_rate': round(fraud_rate, 2),
                    'avg_probability': round(avg_prob, 3),
                    'first_transaction': first,
                    'last_transaction': last
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get total stats: {e}")
            return {}

# Global history manager instance
history_manager = HistoryManager()
