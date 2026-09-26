from docmind.extraction import SCHEMAS, Invoice


def test_invoice_schema_validates_partial_data():
    inv = Invoice.model_validate({"invoice_number": "INV-1", "total": 100.5})
    assert inv.vendor is None
    assert inv.line_items == []


def test_schema_registry_has_expected_types():
    assert set(SCHEMAS) == {"invoice", "resume", "contract"}