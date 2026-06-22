def test_kquant_roundtrip():
    # packs DASH-Q values into Q2_K/Q3_K and asserts gguf-py reads them back
    # with sane reconstruction (byte-layout correctness).
    from dashq.export_kquant import demo
    demo()
