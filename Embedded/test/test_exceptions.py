#cython: language_level=3

#import sys
import pytest
import omniscidbe as dbe
import ctypes
import pandas
import pyarrow
import numpy
import shutil
ctypes._dlopen('libDBEngine.so', ctypes.RTLD_GLOBAL)

data_path = "dbe_test_data"

#######################Check init with wrong parameter
def test_failed_init():
    global engine
    try:
        shutil.rmtree(data_path)
    except FileNotFoundError:
        pass

    with pytest.raises(RuntimeError) as excinfo:
        engine = dbe.PyDbEngine(data='/' + data_path, calcite_port=9091)
    assert "Permission denied" in str(excinfo.value)

######################Check init with right parameters
def test_success_init():
    global engine
    engine = dbe.PyDbEngine(data=data_path, calcite_port=9091)
    assert bool(engine.closed) == False

engine = None

######################Check DDL statement
def test_success_DDL():
    engine.executeDDL("drop table if exists test")
    engine.executeDDL("create table test (x int not null, w tinyint, y int, z text)")
    assert engine.get_tables() == ['test']

#######################Check creating a duplicate table
def test_failed_DDL():
    with pytest.raises(RuntimeError) as excinfo:
        engine.executeDDL("create table test (x int not null, w tinyint, y int, z text)")
    assert "already exists" in str(excinfo.value)

#######################Check right DML statement
def test_success_DML():
    engine.executeDML("insert into test values(55,5,3,'la-la-la')")
    engine.executeDML("insert into test values(66,6,1, 'aa')")
    engine.executeDML("insert into test values(77,7,0,'bb')")
    dframe = engine.select_df("select * from test")
    dforig = pandas.DataFrame({'x': [55,66,77], 'w': [5,6,7], 'y': [3,1,0], 'z': ['la-la-la', 'aa', 'bb']})
    dforig = dforig.astype(dtype= {"x":"int32", "w":"int8","y":"int32", "z":"category"})
    assert dframe.equals(dforig)

#######################Check wrong DML statement
def test_failed_DML():
    with pytest.raises(RuntimeError) as excinfo:
        cursor = engine.executeDML("selectTT * from test")
    assert "SQL Error" in str(excinfo.value)

#######################Check zero division exception
def test_zero_division():
    with pytest.raises(RuntimeError) as excinfo:
        cursor = engine.executeDML("SELECT x / (x - x) FROM test")
    assert "Division by zero" in str(excinfo.value)

#######################Check double init  exception
def test_double_init():
    with pytest.raises(RuntimeError) as excinfo:
        engine = dbe.PyDbEngine()
    assert "already initialized" in str(excinfo.value)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
