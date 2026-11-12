import os  
import cassio  
from dotenv import  load_dotenv

load_dotenv()  ## here we  load the  env varibales  

## This code  connect to the cassandra  
def connect_to_cassandra():
    """
    Initialize connection with Astra DB using CassIO.
    """

    # Get credentials from .env
    application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    database_id = os.getenv("ASTRA_DB_ID")
    keyspace = os.getenv("ASTRA_DB_KEYSPACE")

    # Validate credentials
    if not application_token:
        raise ValueError("ASTRA_DB_APPLICATION_TOKEN is missing")

    if not database_id:
        raise ValueError("ASTRA_DB_ID is missing")

    if not keyspace:
        raise ValueError("ASTRA_DB_KEYSPACE is missing")

    # Initialize CassIO
    cassio.init(
        token=application_token,
        database_id=database_id,
        keyspace=keyspace
    )

    print("Successfully initialized Cassandra connection!")

    return cassio.config.resolve_session()


if __name__ == "__main__":
    session = connect_to_cassandra()

    print("Connected to Astra DB!")
    print(f"Session: {type(session).__name__}")
