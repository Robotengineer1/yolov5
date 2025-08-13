#!/usr/bin/env python3
"""
Database Flush Scripts for YOLO Detection System
Multiple options to clear/reset the PostgreSQL database
"""

import psycopg2
import sys
import os

# Database Configuration - UPDATE THESE WITH YOUR SETTINGS
DB_CONFIG = {
    'host': 'localhost',
    'port': '5432',
    'database': 'yolo_detection',
    'user': 'postgres',
    'password': 'Careyu@2024'  # UPDATE THIS
}

def get_db_connection():
    """Get database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

def flush_detection_data():
    """Option 1: Clear all detection data (keep table structure)"""
    try:
        conn = get_db_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        # Delete all data but keep table structure
        cursor.execute("DELETE FROM detection_results")
        
        # Reset the auto-increment counter
        cursor.execute("ALTER SEQUENCE detection_results_id_seq RESTART WITH 1")
        
        conn.commit()
        row_count = cursor.rowcount
        
        cursor.close()
        conn.close()
        
        print(f"✅ Successfully deleted {row_count} detection records")
        print("✅ Table structure preserved")
        print("✅ ID sequence reset to 1")
        return True
        
    except Exception as e:
        print(f"❌ Error flushing detection data: {e}")
        return False

def drop_and_recreate_table():
    """Option 2: Drop and recreate the detection_results table"""
    try:
        conn = get_db_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        # Drop the table
        cursor.execute("DROP TABLE IF EXISTS detection_results CASCADE")
        print("✅ Table dropped")
        
        # Recreate the table
        cursor.execute('''
            CREATE TABLE detection_results (
                id SERIAL PRIMARY KEY,
                image_name VARCHAR(255) NOT NULL,
                image_path TEXT NOT NULL,
                detection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_detections INTEGER DEFAULT 0,
                present_count INTEGER DEFAULT 0,
                missing_count INTEGER DEFAULT 0,
                image_resolution VARCHAR(50),
                yolo_image_size INTEGER,
                confidence_scores JSONB,
                detection_details JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        print("✅ Table recreated")
        
        # Recreate indexes
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_detection_timestamp 
            ON detection_results(detection_timestamp DESC)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_image_name 
            ON detection_results(image_name)
        ''')
        print("✅ Indexes recreated")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ Table successfully recreated with fresh structure")
        return True
        
    except Exception as e:
        print(f"❌ Error recreating table: {e}")
        return False

def truncate_table():
    """Option 3: TRUNCATE table (fastest way to clear data)"""
    try:
        conn = get_db_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        # Truncate table (faster than DELETE)
        cursor.execute("TRUNCATE TABLE detection_results RESTART IDENTITY CASCADE")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ Table truncated successfully")
        print("✅ All data cleared and ID sequence reset")
        return True
        
    except Exception as e:
        print(f"❌ Error truncating table: {e}")
        return False

def backup_before_flush():
    """Option 4: Backup data before flushing"""
    try:
        conn = get_db_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        # Get all data
        cursor.execute("SELECT * FROM detection_results ORDER BY id")
        results = cursor.fetchall()
        
        if not results:
            print("ℹ️ No data to backup")
            return True
        
        # Save to CSV file
        import csv
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"detection_backup_{timestamp}.csv"
        
        with open(backup_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow([
                'id', 'image_name', 'image_path', 'detection_timestamp',
                'total_detections', 'present_count', 'missing_count',
                'image_resolution', 'yolo_image_size', 'confidence_scores',
                'detection_details', 'created_at', 'updated_at'
            ])
            
            # Write data
            writer.writerows(results)
        
        cursor.close()
        conn.close()
        
        print(f"✅ Backup saved to: {backup_file}")
        print(f"✅ Backed up {len(results)} records")
        return True
        
    except Exception as e:
        print(f"❌ Error creating backup: {e}")
        return False

def show_table_stats():
    """Show current table statistics"""
    try:
        conn = get_db_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        # Get table stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_records,
                MIN(detection_timestamp) as first_detection,
                MAX(detection_timestamp) as last_detection,
                SUM(total_detections) as total_parts_detected,
                SUM(present_count) as total_present,
                SUM(missing_count) as total_missing
            FROM detection_results
        """)
        
        stats = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        print("\n📊 Current Database Statistics:")
        print(f"   Total Records: {stats[0]}")
        print(f"   First Detection: {stats[1]}")
        print(f"   Last Detection: {stats[2]}")
        print(f"   Total Parts Detected: {stats[3]}")
        print(f"   Total Present: {stats[4]}")
        print(f"   Total Missing: {stats[5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        return False

def main():
    """Main function with menu"""
    print("🗄️ YOLO Detection Database Flush Tool")
    print("=====================================")
    
    # Show current stats
    show_table_stats()
    
    print("\nSelect an option:")
    print("1. Clear all detection data (keep table structure)")
    print("2. Drop and recreate table completely")
    print("3. TRUNCATE table (fastest)")
    print("4. Backup data first, then clear")
    print("5. Show current stats only")
    print("6. Exit")
    
    try:
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            print("\n🔄 Clearing all detection data...")
            if flush_detection_data():
                print("✅ Database flushed successfully!")
                
        elif choice == "2":
            confirm = input("⚠️ This will completely drop and recreate the table. Continue? (yes/no): ")
            if confirm.lower() == 'yes':
                print("\n🔄 Dropping and recreating table...")
                if drop_and_recreate_table():
                    print("✅ Table recreated successfully!")
            else:
                print("❌ Operation cancelled")
                
        elif choice == "3":
            print("\n🔄 Truncating table...")
            if truncate_table():
                print("✅ Table truncated successfully!")
                
        elif choice == "4":
            print("\n🔄 Creating backup...")
            if backup_before_flush():
                confirm = input("Backup complete. Proceed with clearing data? (yes/no): ")
                if confirm.lower() == 'yes':
                    if flush_detection_data():
                        print("✅ Database flushed successfully!")
                else:
                    print("❌ Flush cancelled")
                    
        elif choice == "5":
            show_table_stats()
            
        elif choice == "6":
            print("👋 Goodbye!")
            sys.exit(0)
            
        else:
            print("❌ Invalid choice. Please select 1-6.")
            
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()