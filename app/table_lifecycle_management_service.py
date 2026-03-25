from base_service import BaseService
from config import *
from preprocess import *
from connection import pool
import psycopg2

class TableLifecycleManagementService(BaseService):

  def get_connection_hms(self, catalog: str):
    return {
      # "host": "postgres-cluster-pooler-ro.db-service.svc.cluster.local",  # Prod
      "host": "postgres-cluster-pooler-rw.db-service.svc.cluster.local",    # Dev
      "port": 5432,
      "database": f"hms_{catalog}",
      "user": "hms_rw",
      "password": "v3Elc8C9WQObwgm"
      }

  def get_partition_column(self, catalog: str, schema: str, table: str):
    if catalog == 'spark_catalog':
      pass
    else:
      query_partition_column = f"DESCRIBE {catalog}.{schema}.{table}.partitions"
      conn = pool.get()
      cursor = conn.cursor()
      cursor.execute(query_partition_column)
      partition_column = cursor.fetchone()
      if partition_column[0] == 'partition':
        partition_column_information = partition_column[1]
        return partition_column_information.split("<")[1].split(":")[0]
      return None

  def get_table_properties(self, catalog: str, key_property) -> list[tuple[str, str]]:
    DB_CONFIG = self.get_connection_hms(catalog)
    query = f"""
      SELECT d."NAME" as schema
        , t."TBL_NAME" as table
        , p."PARAM_VALUE" as value
      FROM "DBS" d
        JOIN "TBLS" t 
          ON d."DB_ID" = t."DB_ID"
        JOIN "TABLE_PARAMS" p
          ON p."TBL_ID" = t."TBL_ID"
      WHERE "PARAM_KEY" = %s
    """
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(query, (key_property, ))
    tbl_properties = cur.fetchall()
    cur.close()
    conn.close()
    return [(f"{catalog}.{tbl[0]}.{tbl[1]}", tbl[2]) for tbl in tbl_properties]

  def run_batch(self):
    logger.info("gohehehehehe")
    print("hehehe")
    try:
      conn = pool.get()
      cursor = conn.cursor()
      for catalog in ["sandbox"]:
        list_tables_with_expiration_date = self.get_table_properties(catalog, "table-expiration-date")
        for table in list_tables_with_expiration_date:
          if datetime.datetime.now() > table[1]:
            # cursor.execute(f"DROP TABLE {table[0]}") ### Call service insert into recovery table
            cursor.execute(f"SELECT 1")
            logger.info("GO HERE HEHE")
        
        list_tables_with_partition_expiration_days = self.get_table_properties(catalog, "partition-expiration-days")
        list_tables = [f"'{table}'" for table in list_tables_with_partition_expiration_days]
        # str_tables = f"({",".join(list_tables)})"
        str_tables = f"({','.join(list_tables)})"
        query = f"""
          SELECT DISTINCT li.id, split(li.partitioned_by, ',')[0] 
          FROM analytics.lakehouse_storage_management.lakehouse_information li
            JOIN analytics.lakehouse_storage_management.table_metadata tm
              ON li.id = tm.id AND split(li.partitioned_by, ',')[0] = tm.column_name
          WHERE tm.data_type IN ('timestamp', 'date', "timestamp_ntz") 
            AND li.is_existed = True 
            {f"AND id IN {str_tables}" if str_tables != "()" else ""};
        """
        logger.info(query)
        cursor.execute(query)

        list_tables_partitioned = cursor.fetchall() ## chua xu ly case hidden partitioning
        list_tables_partitioned_processed = list_tables_partitioned # [(table_id, col_partitioned.split(",")[0]) for table_id, col_partitioned in list_tables_partitioned]

        dict_tables_partitioned = dict(list_tables_partitioned_processed)

        list_tables_column_partitioned_expiration_days = [
          (table_id, dict_tables_partitioned[table_id], property) 
          for table_id, property in list_tables_with_partition_expiration_days 
          if table_id in dict_tables_partitioned
        ]
        for table_id, partitioned_by, property in list_tables_column_partitioned_expiration_days:
          # cursor.execute(f"""
          #   DELETE FROM {table_id}
          #   WHERE {partitioned_by} = DATE_SUB(CURRENT_DATE(), {property})
          # """)
          logger.info("DELETE PARTITION SUCCESSFULLY")
          logger.info(f"""
            DELETE FROM {table_id}
            WHERE {partitioned_by} = DATE_SUB(CURRENT_DATE(), {property})
          """)
    except Exception as e:
      logger.error(f"[TableLifecycleManagementService] Cannot process phase due to error: {e}")
    finally:
      if cursor:
        cursor.close()
      if conn:
        pool.release(conn)
      logger.info(f"[TableLifecycleManagementService] A cycle is completed.")
        

    
  def run_event(self):
    pass
