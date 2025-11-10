import boto3
import sys

def clear_dynamodb_entries():
    """Clear all entries from the DynamoDB table without deleting the table"""
    try:
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('ServerMetrics')
        
        # Scan all items
        response = table.scan()
        items = response.get('Items', [])
        
        if not items:
            print("ℹ️ No entries found in DynamoDB table")
            return
        
        # Delete all items
        with table.batch_writer() as batch:
            for item in items:
                batch.delete_item(
                    Key={
                        'simulation_id': item['simulation_id'],
                        'server_id': item['server_id']
                    }
                )
        
        print(f"✅ Successfully cleared {len(items)} entries from DynamoDB")
    except Exception as e:
        print(f"❌ Error clearing DynamoDB entries: {str(e)}")

def delete_dynamodb_table():
    """Delete the entire DynamoDB table"""
    try:
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('ServerMetrics')
        
        # Delete table
        table.delete()
        
        # Wait for table to be deleted
        print("⏳ Waiting for table deletion...")
        waiter = dynamodb.meta.client.get_waiter('table_not_exists')
        waiter.wait(TableName='ServerMetrics')
        
        print("✅ Successfully deleted DynamoDB table")
    except dynamodb.meta.client.exceptions.ResourceNotFoundException:
        print("ℹ️ DynamoDB table does not exist")
    except Exception as e:
        print(f"❌ Error deleting DynamoDB table: {str(e)}")

def clear_cloudwatch_metrics():
    """Delete all CloudWatch metrics but keep the dashboard"""
    try:
        cloudwatch = boto3.client('cloudwatch')
        
        # Delete all metrics in the LoadBalancerMetrics namespace
        cloudwatch.delete_metrics(
            Namespace='LoadBalancerMetrics'
        )
        
        print("✅ Successfully cleared CloudWatch metrics")
    except Exception as e:
        print(f"❌ Error clearing CloudWatch metrics: {str(e)}")

def delete_cloudwatch_dashboard():
    """Delete the CloudWatch dashboard"""
    try:
        cloudwatch = boto3.client('cloudwatch')
        
        # Delete the dashboard
        cloudwatch.delete_dashboards(
            DashboardNames=['Load_Balancer_Monitoring']
        )
        
        print("✅ Successfully deleted CloudWatch dashboard")
    except Exception as e:
        print(f"❌ Error deleting CloudWatch dashboard: {str(e)}")

def show_menu():
    print("\n🧹 AWS Cleanup Utilities")
    print("=" * 50)
    print("1. Clear DynamoDB Entries (keeps table structure)")
    print("2. Delete DynamoDB Table")
    print("3. Clear CloudWatch Metrics")
    print("4. Delete CloudWatch Dashboard")
    print("5. Exit")
    print("=" * 50)

def main():
    while True:
        show_menu()
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            confirm = input("⚠️ Are you sure you want to clear all DynamoDB entries? (y/n): ")
            if confirm.lower() == 'y':
                clear_dynamodb_entries()
        
        elif choice == '2':
            confirm = input("⚠️ Are you sure you want to delete the DynamoDB table? (y/n): ")
            if confirm.lower() == 'y':
                delete_dynamodb_table()
        
        elif choice == '3':
            confirm = input("⚠️ Are you sure you want to clear all CloudWatch metrics? (y/n): ")
            if confirm.lower() == 'y':
                clear_cloudwatch_metrics()
        
        elif choice == '4':
            confirm = input("⚠️ Are you sure you want to delete the CloudWatch dashboard? (y/n): ")
            if confirm.lower() == 'y':
                delete_cloudwatch_dashboard()
        
        elif choice == '5':
            print("\n👋 Goodbye!")
            sys.exit(0)
        
        else:
            print("\n❌ Invalid choice. Please enter a number between 1 and 5.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    print("🔧 AWS Cleanup Utility")
    print("This utility helps clean up AWS resources used by the load balancer simulation")
    print("⚠️ Warning: These operations cannot be undone!")
    main()