"""
Test that the root module can be imported
"""


def test_root_import():

    import surrogates

    assert surrogates is not None
