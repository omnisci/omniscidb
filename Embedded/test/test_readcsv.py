import os
import io
import datetime
import pytest
import pyarrow as pa
from pyarrow import csv
import omniscidbe as dbe
import ctypes
ctypes._dlopen('libDBEngine.so', ctypes.RTLD_GLOBAL)

root = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "Tests/Import/datafiles"
)

def test_init():
    global engine
    engine = dbe.PyDbEngine(
            enable_union=1,
            enable_columnar_output=1,
            enable_lazy_fetch=0,
            null_div_by_zero=1,
        )
    assert bool(engine.closed) == False

engine = None

def test_santander():
    table = csv.read_csv(root + "/santander_top1000.csv")
    assert table
    engine.importArrowTable("santander", table)
    assert bool(engine.closed) == False
    r = engine.executeDML("select * from santander")
    assert r
    assert r.colCount() == 202
    assert r.rowCount() == 999

def test_usecols_csv():
    target = {
        'a': [1, 2, 3, 4, 5, 6],
        'b': [2, 3, 4, 5, 6, 7],
        'c': [3, 4, 5, 6, 7, 8],
        'd': [4, 5, 6, 7, 8, 9],
        'e': ['5', '6', '7', '8', '9', '0']
    }
    fp = io.BytesIO(
        b'a,b,c,d,e\n1,2,3,4,5\n2,3,4,5,6\n3,4,5,6,7\n4,5,6,7,8\n5,6,7,8,9\n6,7,8,9,0'
    )
    fp.seek(0)
    table = csv.read_csv(
        fp,
        convert_options=csv.ConvertOptions(
            column_types={
                'a': pa.int32(),
                'b': pa.int64(),
                'c': pa.int64(),
                'd': pa.int64(),
                'e': pa.string(),
            }
        )
    )
    assert table
    engine.importArrowTable("usecols", table)
    assert bool(engine.closed) == False
    cursor = engine.executeDML("select * from usecols")
    assert cursor
    batch = cursor.getArrowRecordBatch()
    assert batch
    assert batch.to_pydict() == target

def test_time_parsing():
    target = {
        'timestamp': [datetime.datetime(2010, 4, 1, 0, 0), datetime.datetime(2010, 4, 1, 0, 30), datetime.datetime(2010, 4, 1, 1, 0)],
        'symbol': ['USD/JPY', 'USD/JPY', 'USD/JPY'],
        'high': [93.526, 93.475, 93.421],
        'low': [93.361, 93.352, 93.326],
        'open': [93.518, 93.385, 93.391],
        'close': [93.382, 93.391, 93.384],
        'spread': [0.005, 0.006, 0.006],
        'volume': [3049, 2251, 1577]
    }
    fp = io.BytesIO(
        b'timestamp,symbol,high,low,open,close,spread,volume\n'
        b'2010-04-01 00:00:00,USD/JPY,93.52600,93.36100,93.51800,93.38200,0.00500,3049\n'
        b'2010-04-01 00:30:00,USD/JPY,93.47500,93.35200,93.38500,93.39100,0.00600,2251\n'
        b'2010-04-01 01:00:00,USD/JPY,93.42100,93.32600,93.39100,93.38400,0.00600,1577\n'
    )
    fp.seek(0)
    table = csv.read_csv(fp)
    assert table
    engine.importArrowTable("time_parsing", table)
    assert bool(engine.closed) == False
    cursor = engine.executeDML("select * from time_parsing")
    assert cursor
    batch = cursor.getArrowRecordBatch()
    assert batch
    assert batch.to_pydict() == target

def test_csv_fillna():
    target = {
        'CRIM': [0.00632],
        'ZN': [18.0],
        'INDUS': [2.31],
        'CHAS': [0.0],
        'NOX': [0.538],
        'RM': [6.575],
        'AGE': [65.2],
        'DIS': [4.09],
        'RAD': [1.0],
        'TAX': [296.0],
        'PTRATIO': [15.3],
        'B': [396.9],
        'LSTAT': [4.98],
        'PRICE': [24.0]
    }
    fp = io.BytesIO(
        b',CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,PRICE\n'
        b'0,0.00632,18.0,2.31,0.0,0.538,6.575,65.2,4.09,1.0,296.0,15.3,396.9,4.98,24.0\n'
    )
    fp.seek(0)
    table = csv.read_csv(fp)
    assert table
    engine.importArrowTable("csv_fillna", table)
    assert bool(engine.closed) == False
    cursor = engine.executeDML("select CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,PRICE from csv_fillna")
    assert cursor
    batch = cursor.getArrowRecordBatch()
    assert batch
    assert batch.to_pydict() == target

def test_null_col():
    target = {'a': [1, 2, 3], 'b': [1, 2, 3], 'c': [None, None, None]}
    fp = io.BytesIO(b'a,b,c\n1,1,\n2,2,\n3,3,\n')
    fp.seek(0)
    table = csv.read_csv(
        fp,
        convert_options=csv.ConvertOptions(
            column_types={
                'a': pa.int32(),
                'b': pa.int64(),
                'c': pa.int64(),
            }
        )
    )
    assert table
    engine.importArrowTable("test_null_col", table)
    assert bool(engine.closed) == False
    cursor = engine.executeDML("select * from test_null_col")
    assert cursor
    batch = cursor.getArrowRecordBatch()
    assert batch
    assert batch.to_pydict() == target


if __name__ == "__main__":
    pytest.main(["-v", __file__])

