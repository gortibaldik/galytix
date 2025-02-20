class Config:
    # these files should be put in the data/ directory 
    save_path: str = "data/GoogleNews-vectors-negative300.bin.gz"
    phrases_path: str = "data/phrases.csv"

    drive_url: str = "https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM"

    # these files are automatically created
    extracted_path: str = "data/vectors.csv"
    extracted_processed_path: str = "data/vectors-processed.csv"
    db_connection_str: str = "postgresql+psycopg2://postgres:postgres@postgres:5432/embedding_engine"