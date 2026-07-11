import psycopg2
from psycopg2.extras import RealDictCursor

# --- Database Connection Settings ---
# Update these credentials to match your local PostgreSQL setup
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "Postgresql@2005",
    "host": "localhost",
    "port": "5432"
}

def get_db_connection():
    """Establishes and returns a secure connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None

# ---------------------------------------------------------
# 1. INSERT RAW TRANSACTION DATA
# ---------------------------------------------------------
def insert_raw_transaction(transaction_data):
    """
    Inserts raw transaction data from the web application into the database.
    """
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        insert_query = """
            INSERT INTO raw_transactions (
                step, customer_id, zipcode_origin, merchant_id, zip_merchant, 
                age, gender, category, amount, fraud
            ) VALUES (
                %(step)s, %(customer_id)s, %(zipcode_origin)s, %(merchant_id)s, %(zip_merchant)s, 
                %(age)s, %(gender)s, %(category)s, %(amount)s, %(fraud)s
            ) RETURNING transaction_id;
        """
        
        # Execute the query using the dictionary keys
        cursor.execute(insert_query, transaction_data)
        transaction_id = cursor.fetchone()[0]
        
        conn.commit()
        cursor.close()
        print(f"Transaction {transaction_id} inserted successfully.")
        return True
        
    except Exception as e:
        print(f"Error inserting transaction: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

# ---------------------------------------------------------
# 2. TRACK NEW ENTRIES & TRIGGER RETRAINING
# ---------------------------------------------------------
def track_and_check_threshold():
    """
    Tracks the number of new entries and triggers retraining when a defined threshold is reached.
    Returns True if the threshold is met (triggering retraining), False otherwise.
    """
    conn = get_db_connection()
    if not conn:
        return False
        
    try:
        cursor = conn.cursor()
        
        # Increment the counter for new records
        update_query = """
            UPDATE pipeline_metadata 
            SET new_records_count = new_records_count + 1 
            WHERE id = 1
            RETURNING new_records_count, retrain_threshold;
        """
        cursor.execute(update_query)
        result = cursor.fetchone()
        
        current_count = result[0]
        threshold = result[1]
        
        conn.commit()
        
        # Check if the threshold is met
        if current_count >= threshold:
            print(f"Threshold reached ({current_count}/{threshold}). Triggering retraining cycle...")
            return True
            
        print(f"Record logged. Current progress to retraining: {current_count}/{threshold}")
        return False
        
    except Exception as e:
        print(f"Error tracking threshold: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()

# ---------------------------------------------------------
# 3. RETRIEVE STORED RECORDS FOR RETRAINING
# ---------------------------------------------------------
def retrieve_unprocessed_records():
    """
    Retrieves stored records for model retraining.
    Returns the data as a list of dictionaries (easy to load into Pandas).
    """
    conn = get_db_connection()
    if not conn:
        return None
        
    try:
        # RealDictCursor returns rows as Python dictionaries instead of tuples
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Fetching records that haven't been pushed to the clean_features table yet
        # (Assuming you add a flag or logic to separate processed vs. unprocessed)
        fetch_query = """
            SELECT * FROM raw_transactions WHERE is_processed = FALSE ORDER BY created_at ASC;
        """
        cursor.execute(fetch_query)
        records = cursor.fetchall()
        
        return records
        
    except Exception as e:
        print(f"Error retrieving records: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def reset_metadata_counter():
    """Resets the record counter back to 0 after a successful retraining cycle."""
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE pipeline_metadata SET new_records_count = 0 WHERE id = 1;")
        conn.commit()
        cursor.close()
        conn.close()
        print("Pipeline metadata counter reset to 0.")