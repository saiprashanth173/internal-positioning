import json
import os
import time

import psycopg2
from psycopg2.extras import DictCursor, RealDictCursor

connection = psycopg2.connect(user=os.getenv('PG_USER'),
                              password=os.getenv('PG_PASSWORD'),
                              host=os.getenv('PG_HOST'),
                              port="5432",
                              database=os.getenv('PG_DB'))


def insert_positions(positions):
    cur = connection.cursor()
    try:
        positions = json.loads(positions)
        inserts_str = []
        for position in positions:
            position["BUILDINGID"] = int(position["BUILDINGID"])
            position["FLOOR"] = int(position["FLOOR"])

            position.pop("TIMESTAMP")
            inserts_str.append(
                "({LONGITUDE}, {LATITUDE}, {FLOOR}, {BUILDINGID}, {SPACEID}, {RELATIVEPOSITION}, {USERID}, {PHONEID})".format(
                    **position))
        inserts = ", ".join(inserts_str)
        sql = "INSERT INTO positions(LONGITUDE, LATITUDE, FLOOR, BUILDINGID, SPACEID, RELATIVEPOSITION, USERID, PHONEID) VALUES {}".format(
            inserts)
        start = time.time()
        cur.execute(sql)
        connection.commit()

        time_str = "Insert time: {} secs\n".format(time.time() - start)
        open("insert_time_file.txt", "a").write(time_str)
        print(time_str)
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error while creating PostgreSQL table", error)
    finally:
        cur.close()


def get_latest_positions():
    dict_cur = connection.cursor(cursor_factory=RealDictCursor)
    dict_cur.execute(
        '''SELECT distinct on (userid) longitude as "LONGITUDE", latitude as "LATITUDE", userid as "USERID", floor as "FLOOR", buildingid as "BUILDINGID", to_char(time_stamp, 'MM-DD-YYYY HH12:MI:SS') as TIMESTAMP, EXTRACT(EPOCH FROM ( NOW() - TIME_STAMP)) > 60 as is_old  FROM positions ORDER BY userid, TIME_STAMP desc;''')
    return [dict(v) for v in dict_cur.fetchall()]
