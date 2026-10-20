# How does a circuit file become input data

`src/data/parse_blknet.py` reads BLOCKS into ordered string IDs with positive finite floating-point widths/heights, then reads NETS as endpoint-name lists. It preserves connectivity without assigning placement coordinates.

```powershell
# Parse the supplied five-macro input and export its actual data.
python steps/00-input/run.py
# Verify all supplied sizes and malformed-input behavior.
python -m unittest discover -s tests/python -p test_parse_blknet.py -v
```

Open chapter 00 in the structured reader or lab to compare the original text and parsed values. Duplicate IDs, malformed dimensions and unknown endpoint references are rejected. The parser does not invent canvas dimensions, pin offsets, rotations or coordinates. Chapter 01 adds an explicit canvas and placement state.
