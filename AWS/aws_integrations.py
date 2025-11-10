import boto3
import pandas as pd
from datetime import datetime
import uuid
import os
from decimal import Decimal

def initialize_aws_services():
    """
    Initialize AWS services (DynamoDB) using boto3.
    """
    try:
        # Create boto3 client for DynamoDB
        dynamodb = boto3.resource('dynamodb')
        print("✅ Successfully connected to AWS services")
        return dynamodb, None
    except Exception as e:
        print(f"❌ Failed to connect to AWS services: {str(e)}")
        return None, None

def create_dynamodb_table(dynamodb):
    """
    Create DynamoDB table for storing server metrics if it doesn't exist.
    """
    table_name = 'ServerMetrics'
    
    try:
        # First try to get the table
        try:
            table = dynamodb.Table(table_name)
            table.table_status  # This will raise an exception if table doesn't exist
            print(f"ℹ️ Table {table_name} already exists")
            return table
        except dynamodb.meta.client.exceptions.ResourceNotFoundException:
            # Table doesn't exist, create it
            table = dynamodb.create_table(
                TableName=table_name,
                KeySchema=[
                    {'AttributeName': 'simulation_id', 'KeyType': 'HASH'},  # Partition key
                    {'AttributeName': 'server_id', 'KeyType': 'RANGE'}  # Sort key
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'simulation_id', 'AttributeType': 'S'},
                    {'AttributeName': 'server_id', 'AttributeType': 'S'}
                ],
                ProvisionedThroughput={
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            )
            # Wait for table to be created
            table.meta.client.get_waiter('table_exists').wait(TableName=table_name)
            print(f"✅ Created DynamoDB table: {table_name}")
            return table
            
    except dynamodb.meta.client.exceptions.ResourceInUseException:
        # Table is being created
        print(f"⏳ Table {table_name} is being created...")
        waiter = dynamodb.meta.client.get_waiter('table_exists')
        waiter.wait(TableName=table_name)
        return dynamodb.Table(table_name)
    except Exception as e:
        print(f"❌ Failed to create DynamoDB table: {str(e)}")
        return None

def clear_previous_dynamodb_entries(table, simulation_id):
    """
    Clear all entries for a given simulation ID from DynamoDB
    """
    try:
        # Query all items for this simulation
        response = table.query(
            KeyConditionExpression='simulation_id = :sid',
            ExpressionAttributeValues={':sid': simulation_id}
        )
        
        # Delete all found items
        with table.batch_writer() as batch:
            for item in response.get('Items', []):
                batch.delete_item(
                    Key={
                        'simulation_id': item['simulation_id'],
                        'server_id': item['server_id']
                    }
                )
        print(f"✅ Cleared previous entries for simulation {simulation_id}")
        return True
    except Exception as e:
        print(f"⚠️ Warning: Failed to clear previous entries: {str(e)}")
        return False

def log_server_metrics_to_dynamodb(table, simulation_id, server_metrics):
    """
    Log server metrics to DynamoDB
    """
    try:
        # Clear previous entries first
        clear_previous_dynamodb_entries(table, simulation_id)
        
        # Sort server metrics by server ID and remove duplicates
        seen_servers = set()
        unique_sorted_metrics = []
        
        for metric in sorted(server_metrics, key=lambda x: int(x['ServerID'].split('-')[1])):
            server_key = (metric['ServerID'], metric['timestamp'] if 'timestamp' in metric else '')
            if server_key not in seen_servers:
                seen_servers.add(server_key)
                unique_sorted_metrics.append(metric)
        
        # Batch write items to DynamoDB
        with table.batch_writer() as batch:
            for metric in unique_sorted_metrics:
                batch.put_item(Item={
                    'simulation_id': simulation_id,
                    'server_id': f"Server-{metric['ServerID']}",
                    'timestamp': str(datetime.now()),
                    'response_time': Decimal(str(metric['ResponseTime'])),
                    'utilization': Decimal(str(metric['Utilization'])),
                    'task_load': Decimal(str(metric['TaskLoad'])),
                    'total_tasks': int(metric['TotalTasks']),
                    'approach': metric['Approach']
                })
        print(f"✅ Successfully logged server metrics to DynamoDB for simulation {simulation_id}")
        return True
    except Exception as e:
        print(f"❌ Failed to log to DynamoDB: {str(e)}")
        return False

def process_csv_and_log_to_aws(csv_file_path, selected_algorithm=None):
    """
    Process the metrics CSV file and log data to AWS services
    """
    try:
        # Initialize AWS services
        dynamodb, _ = initialize_aws_services()
        if not dynamodb:
            return False
        
        # Create DynamoDB table
        table = create_dynamodb_table(dynamodb)
        if not table:
            return False
        
        # Read CSV file
        df = pd.read_csv(csv_file_path)
        if df.empty:
            print("❌ No data found in CSV file")
            return False
        
        # Generate a new simulation ID
        simulation_id = str(uuid.uuid4())
        
        # Convert DataFrame to list of dictionaries for easier processing
        metrics_list = df.to_dict('records')
        
        # If selected_algorithm is provided, update the approach in metrics
        if selected_algorithm:
            for metric in metrics_list:
                metric['Approach'] = selected_algorithm
        
        # Log to DynamoDB
        success = log_server_metrics_to_dynamodb(table, simulation_id, metrics_list)
        if not success:
            return False
        
        # Skip CloudWatch logging since it's now handled in visualization.py
        print("✨ CloudWatch metrics will be logged by visualization.py")
        print(f"✅ Successfully processed and logged data for simulation {simulation_id}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to process and log data: {str(e)}")
        return False

def check_aws_data(simulation_id=None):
    """
    Check the data stored in AWS DynamoDB
    """
    try:
        dynamodb, _ = initialize_aws_services()
        if not dynamodb:
            return
            
        table = dynamodb.Table('ServerMetrics')
        
        if simulation_id:
            # Query specific simulation
            response = table.query(
                KeyConditionExpression='simulation_id = :sid',
                ExpressionAttributeValues={':sid': simulation_id}
            )
        else:
            # Scan all items
            response = table.scan()
        
        if 'Items' in response:
            print(f"\n📊 Found {len(response['Items'])} records")
            for item in response['Items']:
                print(f"\nServer: {item['server_id']}")
                print(f"Simulation: {item['simulation_id']}")
                print(f"Timestamp: {item['timestamp']}")
                print(f"Response Time: {item['response_time']:.3f}s")
                print(f"Utilization: {item['utilization']:.2f}%")
                print(f"Approach: {item['approach']}")
                print("-" * 40)
    except Exception as e:
        print(f"❌ Failed to check AWS data: {str(e)}")
