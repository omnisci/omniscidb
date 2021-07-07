#cython: language_level=3

import pytest
import dbe
import ctypes
import pyarrow as pa

ctypes._dlopen('libDBEngine.so', ctypes.RTLD_GLOBAL)


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

def test_import_table():
    data = [
        pa.array([1, 1, 2, 2, 3]),
        pa.array([None, None, 2, 1, 3]),
        pa.array([3, None, None, 2, 1])
    ]
    table = pa.Table.from_arrays(data, ['F_a', 'F_b', 'F_c'])
    engine.importArrowTable('test', table)
    assert engine.get_tables() == ['test']


def test_projection():
    ra = """execute calcite {"rels": [
{"id": "0", "relOp": "EnumerableTableScan", "table": ["omnisci", "test"], "fieldNames": ["F_a", "F_b", "F_c", "rowid"], "inputs": []}, 
{"id": "1", "relOp": "LogicalProject", "fields": ["F___index__", "F_a", "F_b", "F_c"], "exprs": [{"input": 3}, {"input": 0}, {"input": 1}, {"input": 2}]}, 
{"id": "2", "relOp": "LogicalFilter", "condition": {"op": "CASE", "operands": [{"op": "IS NULL", "operands": [{"input": 1}], "type": {"type": "BOOLEAN", "nullable": true}}, 
{"literal": false, "type": "BOOLEAN", "target_type": "BOOLEAN", "scale": -2147483648, "precision": 1, "type_scale": -2147483648, "type_precision": 1}, 
{"op": "=", "operands": [{"input": 1}, {"literal": 1, "type": "DECIMAL", "target_type": "BIGINT", "scale": 0, "precision": 1, "type_scale": 0, "type_precision": 19}], 
"type": {"type": "BOOLEAN", "nullable": true}}], "type": {"type": "BOOLEAN", "nullable": true}}}, 
{"id": "3", "relOp": "LogicalProject", "fields": ["F___index__", "F_a", "F_b", "F_c"], "exprs": [{"input": 0}, {"input": 1}, {"input": 2}, {"input": 3}]}]}
"""
    target = {'F___index__': [0, 1], 'F_a': [1, 1], 'F_b': [None, None], 'F_c': [3, None]}
    cursor = engine.executeRA(ra)
    batch = cursor.getArrowRecordBatch()
    assert batch.to_pydict() == target

def test_drop():
    ra = """execute calcite {"rels": [
{"id": "0", "relOp": "EnumerableTableScan", "table": ["omnisci", "test"], "fieldNames": ["F_a", "F_b", "F_c", "rowid"], "inputs": []}, 
{"id": "1", "relOp": "LogicalProject", "fields": ["F_a", "F_b", "F_c"], "exprs": [{"input": 0}, {"input": 1}, {"input": 2}]}, 
{"id": "2", "relOp": "LogicalProject", "fields": ["F_a", "F_b", "F_c"], "exprs": [{"input": 0}, {"input": 1}, {"input": 2}]}, 
{"id": "3", "relOp": "LogicalAggregate", "fields": ["F_a", "F_b", "F_c"], "group": [0], "aggs": [{"agg": "SUM", "operands": [1], 
"distinct": false, "type": {"type": "DOUBLE", "nullable": true}}, {"agg": "SUM", "operands": [2], "distinct": false, "type": {"type": "DOUBLE", "nullable": true}}]}, 
{"id": "4", "relOp": "LogicalSort", "collation": [{"field": 0, "direction": "ASCENDING", "nulls": "LAST"}]}, 
{"id": "5", "relOp": "LogicalFilter", "condition": {"op": "CASE", "operands": [{"op": "IS NULL", "operands": [{"input": 1}], "type": {"type": "BOOLEAN", "nullable": true}}, 
{"literal": false, "type": "BOOLEAN", "target_type": "BOOLEAN", "scale": -2147483648, "precision": 1, "type_scale": -2147483648, "type_precision": 1}, 
{"op": ">", "operands": [{"input": 1}, {"literal": 1, "type": "DECIMAL", "target_type": "BIGINT", "scale": 0, "precision": 1, "type_scale": 0, "type_precision": 19}], 
"type": {"type": "BOOLEAN", "nullable": true}}], "type": {"type": "BOOLEAN", "nullable": true}}}, {"id": "6", "relOp": "LogicalProject", 
"fields": ["F_a", "F_b", "F_c"], "exprs": [{"input": 0}, {"input": 1}, {"input": 2}]}]}
"""
    target = {'F_a': [2, 3], '$f1': [3, 3], '$f2': [2, 1]}
    cursor = engine.executeRA(ra)
    batch = cursor.getArrowRecordBatch()
    assert batch.to_pydict() == target

def test_iloc():
    ra = """execute calcite {"rels": [
{"id": "0", "relOp": "EnumerableTableScan", "table": ["omnisci", "test"], "fieldNames": ["F_a", "F_b", "F_c", "rowid"], "inputs": []}, 
{"id": "1", "relOp": "LogicalProject", "fields": ["F___index__", "F_a", "F_b", "F_c"], "exprs": [{"input": 3}, {"input": 0}, {"input": 1}, {"input": 2}]}, 
{"id": "2", "relOp": "LogicalFilter", "condition": {"op": "CASE", "operands": [{"op": "IS NULL", "operands": [{"op": "+", "operands": [{"input": 1}, 
{"op": "+", "operands": [{"input": 2}, {"literal": 2, "type": "DECIMAL", "target_type": "BIGINT", "scale": 0, "precision": 1, "type_scale": 0, "type_precision": 19}], 
"type": {"type": "DOUBLE", "nullable": true}}], "type": {"type": "DOUBLE", "nullable": true}}], "type": {"type": "BOOLEAN", "nullable": true}}, 
{"literal": false, "type": "BOOLEAN", "target_type": "BOOLEAN", "scale": -2147483648, "precision": 1, "type_scale": -2147483648, "type_precision": 1}, 
{"op": ">", "operands": [{"op": "+", "operands": [{"input": 1}, 
{"op": "+", "operands": [{"input": 2}, {"literal": 2, "type": "DECIMAL", "target_type": "BIGINT", "scale": 0, "precision": 1, "type_scale": 0, "type_precision": 19}], 
"type": {"type": "DOUBLE", "nullable": true}}], "type": {"type": "DOUBLE", "nullable": true}}, 
{"literal": 1, "type": "DECIMAL", "target_type": "BIGINT", "scale": 0, "precision": 1, "type_scale": 0, "type_precision": 19}], 
"type": {"type": "BOOLEAN", "nullable": true}}], "type": {"type": "BOOLEAN", "nullable": true}}}, 
{"id": "3", "relOp": "LogicalProject", "fields": ["F___index__", "F_a", "F_b", "F_c"], "exprs": [{"input": 0}, 
{"op": "+", "operands": [{"input": 1}, {"literal": 2, "type": "DECIMAL", "target_type": "BIGINT", "scale": 0, "precision": 1, "type_scale": 0, "type_precision": 19}], 
"type": {"type": "BIGINT", "nullable": true}}, 
{"op": "+", "operands": [{"input": 2}, {"literal": 2, "type": "DECIMAL", "target_type": "BIGINT", "scale": 0, "precision": 1, "type_scale": 0, "type_precision": 19}], 
"type": {"type": "DOUBLE", "nullable": true}}, 
{"op": "+", "operands": [{"input": 3}, {"literal": 2, "type": "DECIMAL", "target_type": "BIGINT", "scale": 0, "precision": 1, "type_scale": 0, "type_precision": 19}], 
"type": {"type": "DOUBLE", "nullable": true}}]}]}
"""
    target = {'F___index__': [2, 3, 4], 'F_a': [4, 4, 5], 'F_b': [4, 3, 5], 'F_c': [None, 4, 3]}
    cursor = engine.executeRA(ra)
    batch = cursor.getArrowRecordBatch()
    assert batch.to_pydict() == target

def test_empty():
    ra = """execute calcite {"rels": [
{"id": "0", "relOp": "EnumerableTableScan", "table": ["omnisci", "test"], "fieldNames": ["F_a", "F_b", "F_c", "rowid"], "inputs": []}, 
{"id": "1", "relOp": "LogicalProject", "fields": ["F___index__", "F_a", "F_b", "F_c"], "exprs": [{"input": 3}, {"input": 0}, {"input": 1}, {"input": 2}]}, 
{"id": "2", "relOp": "LogicalFilter", "condition": {"op": "CASE", "operands": [{"op": "IS NULL", "operands": [{"input": 1}], "type": {"type": "BOOLEAN", "nullable": true}}, 
{"literal": false, "type": "BOOLEAN", "target_type": "BOOLEAN", "scale": -2147483648, "precision": 1, "type_scale": -2147483648, "type_precision": 1}, 
{"op": "=", "operands": [{"input": 1}, {"literal": 1, "type": "DECIMAL", "target_type": "BIGINT", "scale": 0, "precision": 1, "type_scale": 0, "type_precision": 19}], 
"type": {"type": "BOOLEAN", "nullable": true}}], "type": {"type": "BOOLEAN", "nullable": true}}}, 
{"id": "3", "relOp": "LogicalProject", "fields": ["F___index__", "F_a", "F_b", "F_c"], "exprs": [{"input": 0}, {"input": 1}, {"input": 2}, {"input": 3}]}]}
"""
    target = {'F___index__': [0, 1], 'F_a': [1, 1], 'F_b': [None, None], 'F_c': [3, None]}
    cursor = engine.executeRA(ra)
    batch = cursor.getArrowRecordBatch()
    assert batch.to_pydict() == target

def test_filter():
    ra = """execute calcite {"rels": [
{"id": "0", "relOp": "EnumerableTableScan", "table": ["omnisci", "test"], "fieldNames": ["F_a", "F_b", "F_c", "rowid"], "inputs": []}, 
{"id": "1", "relOp": "LogicalProject", "fields": ["F_a", "F_b", "F_c"], "exprs": [{"input": 0}, {"input": 1}, {"input": 2}]}, 
{"id": "2", "relOp": "LogicalProject", "fields": ["F_a", "F_b", "F_c"], "exprs": [{"input": 0}, {"input": 1}, {"input": 2}]}, 
{"id": "3", "relOp": "LogicalAggregate", "fields": ["F_a", "F_b", "F_c"], "group": [0], "aggs": [
{"agg": "SUM", "operands": [1], "distinct": false, "type": {"type": "DOUBLE", "nullable": true}}, 
{"agg": "SUM", "operands": [2], "distinct": false, "type": {"type": "DOUBLE", "nullable": true}}]}, 
{"id": "4", "relOp": "LogicalSort", "collation": [{"field": 0, "direction": "ASCENDING", "nulls": "LAST"}]}, 
{"id": "5", "relOp": "LogicalFilter", "condition": {"op": "CASE", "operands": [{"op": "IS NULL", "operands": [{"input": 1}], "type": {"type": "BOOLEAN", "nullable": true}}, 
{"literal": false, "type": "BOOLEAN", "target_type": "BOOLEAN", "scale": -2147483648, "precision": 1, "type_scale": -2147483648, "type_precision": 1}, 
{"op": ">", "operands": [{"input": 1}, {"literal": 1, "type": "DECIMAL", "target_type": "BIGINT", "scale": 0, "precision": 1, "type_scale": 0, "type_precision": 19}], 
"type": {"type": "BOOLEAN", "nullable": true}}], "type": {"type": "BOOLEAN", "nullable": true}}}, 
{"id": "6", "relOp": "LogicalProject", "fields": ["F_a", "F_b", "F_c"], "exprs": [{"input": 0}, {"input": 1}, {"input": 2}]}]}
"""
    target = {'F_a': [2, 3], '$f1': [3, 3], '$f2': [2, 1]}
    cursor = engine.executeRA(ra)
    batch = cursor.getArrowRecordBatch()
    assert batch.to_pydict() == target

def test_filter_with_index():
    ra = """execute calcite {"rels": [
{"id": "0", "relOp": "EnumerableTableScan", "table": ["omnisci", "test"], "fieldNames": ["F_a", "F_b", "F_c", "rowid"], "inputs": []}, 
{"id": "1", "relOp": "LogicalProject", "fields": ["F___index__", "F_a", "F_b", "F_c"], "exprs": [{"input": 3}, {"input": 0}, {"input": 1}, {"input": 2}]}, 
{"id": "2", "relOp": "LogicalFilter", "condition": {"op": "CASE", "operands": [{"op": "IS NULL", "operands": [{"op": "+", "operands": [{"input": 1}, 
{"op": "+", "operands": [{"input": 2}, {"literal": 2, "type": "DECIMAL", "target_type": "BIGINT", "scale": 0, "precision": 1, "type_scale": 0, "type_precision": 19}], 
"type": {"type": "DOUBLE", "nullable": true}}], "type": {"type": "DOUBLE", "nullable": true}}], "type": {"type": "BOOLEAN", "nullable": true}}, 
{"literal": false, "type": "BOOLEAN", "target_type": "BOOLEAN", "scale": -2147483648, "precision": 1, "type_scale": -2147483648, "type_precision": 1}, 
{"op": ">", "operands": [{"op": "+", "operands": [{"input": 1}, {"op": "+", "operands": [{"input": 2}, 
{"literal": 2, "type": "DECIMAL", "target_type": "BIGINT", "scale": 0, "precision": 1, "type_scale": 0, "type_precision": 19}], 
"type": {"type": "DOUBLE", "nullable": true}}], "type": {"type": "DOUBLE", "nullable": true}}, 
{"literal": 1, "type": "DECIMAL", "target_type": "BIGINT", "scale": 0, "precision": 1, "type_scale": 0, "type_precision": 19}], 
"type": {"type": "BOOLEAN", "nullable": true}}], "type": {"type": "BOOLEAN", "nullable": true}}}, 
{"id": "3", "relOp": "LogicalProject", "fields": ["F___index__", "F_a", "F_b", "F_c"], "exprs": [{"input": 0}, 
{"op": "+", "operands": [{"input": 1}, {"literal": 2, "type": "DECIMAL", "target_type": "BIGINT", "scale": 0, "precision": 1, "type_scale": 0, "type_precision": 19}], "type": {"type": "BIGINT", "nullable": true}}, 
{"op": "+", "operands": [{"input": 2}, {"literal": 2, "type": "DECIMAL", "target_type": "BIGINT", "scale": 0, "precision": 1, "type_scale": 0, "type_precision": 19}], "type": {"type": "DOUBLE", "nullable": true}}, 
{"op": "+", "operands": [{"input": 3}, {"literal": 2, "type": "DECIMAL", "target_type": "BIGINT", "scale": 0, "precision": 1, "type_scale": 0, "type_precision": 19}], "type": {"type": "DOUBLE", "nullable": true}}]}]}
"""
    target = {'F___index__': [2, 3, 4], 'F_a': [4, 4, 5], 'F_b': [4, 3, 5], 'F_c': [None, 4, 3]}
    cursor = engine.executeRA(ra)
    batch = cursor.getArrowRecordBatch()
    assert batch.to_pydict() == target

def test_filter_proj():
    ra = """execute calcite {"rels": [
{"id": "0", "relOp": "EnumerableTableScan", "table": ["omnisci", "test"], "fieldNames": ["F_a", "F_b", "F_c", "rowid"], "inputs": []}, 
{"id": "1", "relOp": "LogicalProject", "fields": ["F___index__", "F_a", "F_b", "F_c"], "exprs": [{"input": 3}, {"input": 0}, {"input": 1}, {"input": 2}]}, 
{"id": "2", "relOp": "LogicalFilter", "condition": {"op": "CASE", "operands": [{"op": "IS NULL", "operands": [{"input": 1}], "type": {"type": "BOOLEAN", "nullable": true}}, 
{"literal": true, "type": "BOOLEAN", "target_type": "BOOLEAN", "scale": -2147483648, "precision": 1, "type_scale": -2147483648, "type_precision": 1}, 
{"op": "<>", "operands": [{"input": 1}, {"literal": 1, "type": "DECIMAL", "target_type": "BIGINT", "scale": 0, "precision": 1, "type_scale": 0, "type_precision": 19}], 
"type": {"type": "BOOLEAN", "nullable": true}}], "type": {"type": "BOOLEAN", "nullable": true}}}, 
{"id": "3", "relOp": "LogicalProject", "fields": ["F___index__", "F_a", "F_b"], 
"exprs": [{"input": 0}, {"op": "*", "operands": [{"input": 1}, {"input": 2}], "type": {"type": "DOUBLE", "nullable": true}}, {"input": 2}]}]}
"""
    target = {'F___index__': [2, 3, 4], 'F_a': [4, 2, 9], 'F_b': [2, 1, 3]}
    cursor = engine.executeRA(ra)
    batch = cursor.getArrowRecordBatch()
    assert batch.to_pydict() == target

def test_filter_drop():
    ra = """execute calcite {"rels": [
{"id": "0", "relOp": "EnumerableTableScan", "table": ["omnisci", "test"], "fieldNames": ["F_a", "F_b", "F_c", "rowid"], "inputs": []}, 
{"id": "1", "relOp": "LogicalProject", "fields": ["F___index__", "F_a", "F_b", "F_c"], "exprs": [{"input": 3}, {"input": 0}, {"input": 1}, {"input": 2}]}, 
{"id": "2", "relOp": "LogicalFilter", "condition": {"op": "CASE", "operands": [{"op": "IS NULL", "operands": [{"input": 1}], "type": {"type": "BOOLEAN", "nullable": true}}, 
{"literal": true, "type": "BOOLEAN", "target_type": "BOOLEAN", "scale": -2147483648, "precision": 1, "type_scale": -2147483648, "type_precision": 1}, 
{"op": "<>", "operands": [{"input": 1}, {"literal": 1, "type": "DECIMAL", "target_type": "BIGINT", "scale": 0, "precision": 1, "type_scale": 0, "type_precision": 19}], 
"type": {"type": "BOOLEAN", "nullable": true}}], "type": {"type": "BOOLEAN", "nullable": true}}}, 
{"id": "3", "relOp": "LogicalProject", "fields": ["F___index__", "F_a", "F_b"], "exprs": [{"input": 0}, 
{"op": "*", "operands": [{"input": 1}, {"input": 2}], "type": {"type": "DOUBLE", "nullable": true}}, {"input": 2}]}]}
"""
    target = {'F___index__': [2, 3, 4], 'F_a': [4, 2, 9], 'F_b': [2, 1, 3]}
    cursor = engine.executeRA(ra)
    batch = cursor.getArrowRecordBatch()
    assert batch.to_pydict() == target

if __name__ == "__main__":
    pytest.main(["-v", __file__])
