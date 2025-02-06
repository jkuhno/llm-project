from langgraph.store.postgres import PostgresStore # type: ignore
from typing_extensions import TypedDict # type: ignore
import os

# Import from langgraph.store.postgres gives an error, manual implementation
class PoolConfig(TypedDict, total=False):
    """Connection pool settings for PostgreSQL connections.

    Controls connection lifecycle and resource utilization:
    - Small pools (1-5) suit low-concurrency workloads
    - Larger pools handle concurrent requests but consume more resources
    - Setting max_size prevents resource exhaustion under load
    """

    min_size: int
    """Minimum number of connections maintained in the pool. Defaults to 1."""

    max_size: int
    """Maximum number of connections allowed in the pool. None means unlimited."""

    kwargs: dict
    """Additional connection arguments passed to each connection in the pool.
    
    Default kwargs set automatically:
    - autocommit: True
    - prepare_threshold: 0
    - row_factory: dict_row
    """


class ConnectPostgres:
    def __init__(self, embeddings, dims):
        user = os.environ['POSTGRES_USER']
        pw = os.environ['POSTGRES_PASSWORD']
        connection_string=f"postgresql://{user}:{pw}@db:5432/postgres"

        self.connection_string = connection_string
        self.embeddings = embeddings
        self.dims = dims
        self.user = user
        self.pw = pw


    def get_store(self):
        """
        Create and return a PostgresStore instance.

        Returns:
            PostgresStore: Configured PostgresStore instance.
        """
        return PostgresStore.from_conn_string(
            self.connection_string,
            pool_config=PoolConfig(
                min_size=5,
                max_size=20
            ),
            index={
                "dims": self.dims,
                "embed": self.embeddings,
            },
        )