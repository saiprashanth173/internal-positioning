    CREATE TABLE IF NOT EXISTS positions
        (
            LONGITUDE DOUBLE PRECISION,
            LATITUDE DOUBLE  PRECISION,
            FLOOR INT,
            BUILDINGID INT,
            SPACEID INT,
            RELATIVEPOSITION INT,
            USERID INT,
            PHONEID INT,
            TIME_STAMP TIMESTAMP DEFAULT NOW()
        );

    CREATE INDEX time_user ON positions (USERID, TIME_STAMP DESC);