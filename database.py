"""
Database configuration and connection management for RFP Extraction Platform.

This module handles all database operations including connection setup,
session management, and database initialization with pgvector support.
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator
import logging

from models import Base, add_constraints

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration from environment variables
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite:///./rfp_extraction.db"  # Use SQLite for local development
)

# For development, you can use SQLite with a fallback
if DATABASE_URL.startswith("sqlite"):
    # SQLite configuration for local development
    engine = create_engine(
        DATABASE_URL,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=True  # Set to False in production
    )
else:
    # PostgreSQL configuration for production
    engine = create_engine(
        DATABASE_URL,
        pool_size=20,
        max_overflow=30,
        pool_pre_ping=True,
        echo=False  # Set to True for debugging
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function for FastAPI to get database sessions.
    
    This function provides a database session that is automatically
    closed after the request completes, ensuring proper resource management.
    
    Yields:
        SQLAlchemy Session: Database session for the current request
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    
    This provides a clean way to manage database sessions with automatic
    cleanup, useful for background tasks and batch operations.
    
    Yields:
        SQLAlchemy Session: Database session
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        db.close()


def init_database() -> None:
    """
    Initialize the database with all tables and extensions.
    
    This function creates all tables defined in the models and sets up
    the pgvector extension for PostgreSQL databases. It should be called
    once during application startup.
    """
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
        # Add constraints if using PostgreSQL
        if not DATABASE_URL.startswith("sqlite"):
            with get_db_session() as db:
                # Enable pgvector extension
                try:
                    db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    logger.info("pgvector extension enabled")
                except Exception as e:
                    logger.warning(f"Could not enable pgvector extension: {e}")
                
                # Add custom constraints
                try:
                    add_constraints()
                    db.commit()
                    logger.info("Database constraints added successfully")
                except Exception as e:
                    logger.warning(f"Could not add constraints: {e}")
        
        logger.info("Database initialization completed")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


def check_database_connection() -> bool:
    """
    Check if the database connection is working.
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        with get_db_session() as db:
            db.execute(text("SELECT 1"))
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def get_database_info() -> dict:
    """
    Get information about the current database configuration.
    
    Returns:
        dict: Database configuration information
    """
    return {
        "database_url": DATABASE_URL.replace(DATABASE_URL.split('@')[0].split('://')[1], "***") if '@' in DATABASE_URL else DATABASE_URL,
        "engine_name": engine.name,
        "pool_size": engine.pool.size() if hasattr(engine.pool, 'size') else "N/A",
        "checked_out_connections": engine.pool.checkedout() if hasattr(engine.pool, 'checkedout') else "N/A"
    }


class DatabaseManager:
    """
    Database manager class for advanced database operations.
    
    This class provides methods for database maintenance, backup operations,
    and performance monitoring specific to the RFP extraction platform.
    """
    
    def __init__(self):
        self.engine = engine
        self.session_factory = SessionLocal
    
    def get_table_stats(self) -> dict:
        """
        Get statistics about database tables.
        
        Returns:
            dict: Table statistics including row counts and sizes
        """
        stats = {}
        
        if DATABASE_URL.startswith("sqlite"):
            # SQLite statistics
            with get_db_session() as db:
                tables = ['documents', 'text_chunks', 'requirements', 'cross_references', 'processing_jobs']
                for table in tables:
                    try:
                        result = db.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        count = result.scalar()
                        stats[table] = {"row_count": count}
                    except Exception as e:
                        stats[table] = {"error": str(e)}
        else:
            # PostgreSQL statistics
            with get_db_session() as db:
                query = text("""
                    SELECT 
                        schemaname,
                        tablename,
                        n_tup_ins as inserts,
                        n_tup_upd as updates,
                        n_tup_del as deletes,
                        n_live_tup as live_rows,
                        n_dead_tup as dead_rows
                    FROM pg_stat_user_tables 
                    WHERE schemaname = 'public'
                    ORDER BY tablename
                """)
                
                result = db.execute(query)
                for row in result:
                    stats[row.tablename] = {
                        "inserts": row.inserts,
                        "updates": row.updates,
                        "deletes": row.deletes,
                        "live_rows": row.live_rows,
                        "dead_rows": row.dead_rows
                    }
        
        return stats
    
    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """
        Clean up expired user sessions.
        
        Args:
            days_old: Number of days after which sessions are considered expired
            
        Returns:
            int: Number of sessions cleaned up
        """
        with get_db_session() as db:
            from models import UserSession
            from datetime import datetime, timedelta
            
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            # Delete expired sessions
            deleted_count = db.query(UserSession).filter(
                UserSession.expires_at < cutoff_date
            ).delete()
            
            logger.info(f"Cleaned up {deleted_count} expired sessions")
            return deleted_count
    
    def vacuum_database(self) -> None:
        """
        Perform database maintenance (VACUUM for PostgreSQL).
        
        This should be run periodically to maintain database performance.
        """
        if not DATABASE_URL.startswith("sqlite"):
            with get_db_session() as db:
                try:
                    db.execute(text("VACUUM ANALYZE"))
                    logger.info("Database vacuum completed")
                except Exception as e:
                    logger.error(f"Database vacuum failed: {e}")
                    raise
        else:
            logger.info("VACUUM not needed for SQLite")


# Global database manager instance
db_manager = DatabaseManager()


# Health check functions for monitoring
def health_check() -> dict:
    """
    Comprehensive health check for the database system.
    
    Returns:
        dict: Health status information
    """
    health_status = {
        "database_connection": False,
        "tables_exist": False,
        "pgvector_available": False,
        "error": None
    }
    
    try:
        # Check basic connection
        health_status["database_connection"] = check_database_connection()
        
        if health_status["database_connection"]:
            with get_db_session() as db:
                # Check if tables exist
                if not DATABASE_URL.startswith("sqlite"):
                    result = db.execute(text("""
                        SELECT COUNT(*) 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name IN ('documents', 'requirements', 'text_chunks')
                    """))
                    table_count = result.scalar()
                    health_status["tables_exist"] = table_count >= 3
                    
                    # Check pgvector extension
                    result = db.execute(text("""
                        SELECT EXISTS(
                            SELECT 1 FROM pg_extension WHERE extname = 'vector'
                        )
                    """))
                    health_status["pgvector_available"] = result.scalar()
                else:
                    # SQLite checks
                    result = db.execute(text("""
                        SELECT COUNT(*) FROM sqlite_master 
                        WHERE type='table' AND name IN ('documents', 'requirements', 'text_chunks')
                    """))
                    table_count = result.scalar()
                    health_status["tables_exist"] = table_count >= 3
                    health_status["pgvector_available"] = True  # Not applicable for SQLite
    
    except Exception as e:
        health_status["error"] = str(e)
        logger.error(f"Health check failed: {e}")
    
    return health_status


# Initialize database on module import (for development)
if __name__ == "__main__":
    # This allows running the database module directly for initialization
    print("Initializing database...")
    init_database()
    print("Database initialization complete!")
    
    # Run health check
    health = health_check()
    print(f"Health check results: {health}")
