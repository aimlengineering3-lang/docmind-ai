import logging
from typing import Literal

from pydantic import BaseModel, Field

log = logging.getLogger("docmind.extraction")


class LineItem(BaseModel):
    description: str
    quantity: float | None = None
    unit_price: float | None = None
    amount: float | None = None


class Invoice(BaseModel):
    invoice_number: str | None = None
    vendor: str | None = None
    date: str | None = None
    total: float | None = None
    currency: str | None = None
    line_items: list[LineItem] = Field(default_factory=list)


class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    skills: list[str] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)


class Contract(BaseModel):
    parties: list[str] = Field(default_factory=list)
    effective_date: str | None = None
    termination_terms: str | None = None
    payment_terms: str | None = None


SCHEMAS: dict[str, type[BaseModel]] = {
    "invoice": Invoice,
    "resume": Resume,
    "contract": Contract,
}

DocType = Literal["invoice", "resume", "contract"]