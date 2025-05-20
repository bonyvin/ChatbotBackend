from sqlalchemy import String, Column, Integer, ForeignKey, Date, Float, Boolean, Table, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.mssql import VARCHAR, INTEGER, DATE, FLOAT, BIT  # Import MSSQL-specific types

Base = declarative_base()  # Use the declarative base

promotion_store_association = Table(
    "promotion_store_association",
    Base.metadata,
    Column("promotionId", VARCHAR(255), ForeignKey("promotionHeader.promotionId"), primary_key=True),
    Column("storeId", VARCHAR(255), ForeignKey("storeDetails.storeId"), primary_key=True)
)

invoice_store_association = Table(
    "invoice_store_association",
    Base.metadata,
    Column("invoiceId", VARCHAR(255), ForeignKey("invoiceHeader.invoiceId"), primary_key=True),
    Column("storeId", VARCHAR(255), ForeignKey("storeDetails.storeId"), primary_key=True)
)

# STORE
class StoreDetails(Base):
    __tablename__ = "storeDetails"
    storeId = Column(VARCHAR(255), primary_key=True, index=True)
    storeName = Column(VARCHAR(255), nullable=False)
    address = Column(VARCHAR(255), nullable=False)
    city = Column(VARCHAR(100), nullable=False)
    state = Column(VARCHAR(100), nullable=False)
    zipCode = Column(VARCHAR(20), nullable=False)
    phone = Column(VARCHAR(20), nullable=False)

    shipmentHeader = relationship("ShipmentHeader", back_populates="store", uselist=False)
    promotions = relationship("PromotionHeader", secondary=promotion_store_association, back_populates="stores")
    invoices = relationship("InvHeader", secondary=invoice_store_association, back_populates="stores")

# PROMOTION
class PromotionHeader(Base):
    __tablename__ = "promotionHeader"

    promotionId = Column(VARCHAR(255), primary_key=True, index=True)
    componentId = Column(VARCHAR(255), nullable=False, index=True, unique=True)
    startDate = Column(DATE, index=True)
    endDate = Column(DATE, index=True)
    promotionType = Column(VARCHAR(20), nullable=False)

    stores = relationship("StoreDetails", secondary=promotion_store_association, back_populates="promotions")

    promotionDetailsbp = relationship(
        "PromotionDetails",
        foreign_keys="[PromotionDetails.promotionId]",
        back_populates="promotionHeaders"
    )

class PromotionDetails(Base):
    __tablename__ = "promotionDetails"

    id = Column(INTEGER, primary_key=True, autoincrement=True)
    promotionId = Column(VARCHAR(255), ForeignKey("promotionHeader.promotionId"))
    componentId = Column(VARCHAR(255), ForeignKey("promotionHeader.componentId"))
    itemId = Column(VARCHAR(255), ForeignKey("itemMaster.itemId"))
    discountType = Column(VARCHAR(255), index=True)
    discountValue = Column(FLOAT, nullable=False)

    promotionHeaders = relationship(
        "PromotionHeader",
        foreign_keys=[promotionId],  # Specify foreign key
        back_populates="promotionDetailsbp"
    )
    itemMaster = relationship("ItemMaster", back_populates="itemPromotionDetail")

# ITEM
class ItemMaster(Base):
    __tablename__ = "itemMaster"

    itemId = Column(VARCHAR(255), primary_key=True, index=True)
    itemDescription = Column(VARCHAR(255), nullable=False)
    itemSecondaryDescription = Column(VARCHAR(255), nullable=False)
    itemDepartment = Column(VARCHAR(20), nullable=False)
    itemClass = Column(VARCHAR(255), nullable=False)
    itemSubClass = Column(VARCHAR(255), nullable=False)
    brand = Column(VARCHAR(255), nullable=False)
    itemSuppliers = relationship("ItemSupplier", back_populates="itemMaster")
    diffType1 = Column(INTEGER, ForeignKey("itemDiffs.id"))
    diffType2 = Column(INTEGER, ForeignKey("itemDiffs.id"))
    diffType3 = Column(INTEGER, ForeignKey("itemDiffs.id"))
    itemDiffs1 = relationship("ItemDiffs", foreign_keys=[diffType1])
    itemDiffs2 = relationship("ItemDiffs", foreign_keys=[diffType2])
    itemDiffs3 = relationship("ItemDiffs", foreign_keys=[diffType3])
    itemPromotionDetail = relationship("PromotionDetails", back_populates="itemMaster")
    itemPurchaseOrder = relationship("PoDetails", back_populates="itemMaster")

class ItemSupplier(Base):
    __tablename__ = "itemSupplier"

    id = Column(INTEGER, primary_key=True, index=True, autoincrement=True)
    supplierCost = Column(FLOAT, nullable=False)
    itemMaster = relationship("ItemMaster", back_populates="itemSuppliers")
    supplier = relationship("Supplier", back_populates="itemSuppliers")
    supplierId = Column(VARCHAR(225), ForeignKey("suppliers.supplierId"))
    itemId = Column(VARCHAR(225), ForeignKey("itemMaster.itemId"))

class ItemDiffs(Base):
    __tablename__ = "itemDiffs"

    id = Column(INTEGER, primary_key=True, index=True, autoincrement=True)
    diffType = Column(VARCHAR(255), nullable=False)
    diffId = Column(VARCHAR(255), nullable=False)
    # itemMaster=relationship("ItemMaster", back_populates="itemDiffs")

# SUPPLIER
class Supplier(Base):
    __tablename__ = "suppliers"

    supplierId = Column(VARCHAR(255), primary_key=True, index=True)
    name = Column(VARCHAR(255), unique=True, nullable=False)
    email = Column(VARCHAR(255), unique=True, nullable=False)
    phone = Column(VARCHAR(20), nullable=False)
    address = Column(VARCHAR(255), nullable=False)
    lead_time = Column(VARCHAR(255), nullable=False)

    poDetails = relationship("PoDetails", back_populates="suppliers")
    itemSuppliers = relationship("ItemSupplier", back_populates="supplier")

# SHIPMENT
class ShipmentHeader(Base):
    __tablename__ = "shipmentHeader"

    receiptId = Column(VARCHAR(255), primary_key=True, index=True)
    asnId = Column(VARCHAR(225), index=True)
    expectedDate = Column(DATE, index=True)
    receivedDate = Column(DATE, index=True)
    receivedBy = Column(VARCHAR(225), index=True)
    status = Column(VARCHAR(225), index=True)
    sourceLocation = Column(VARCHAR(225), index=True)
    destinationLocation = Column(VARCHAR(255), ForeignKey("storeDetails.storeId"))
    totalQuantity = Column(INTEGER, index=True)
    totalCost = Column(FLOAT, index=True)
    poHeader = relationship("PoHeader", back_populates="shipmentHeader")

    poId = Column(VARCHAR(225), ForeignKey("poHeader.poNumber"))
    store = relationship("StoreDetails", back_populates="shipmentHeader")
    shipmentDetails = relationship("ShipmentDetails", back_populates="shipmentHeader")

class ShipmentDetails(Base):
    __tablename__ = 'shipmentDetails'
    id = Column(INTEGER, primary_key=True, index=True, autoincrement=True)
    itemId = Column(VARCHAR(225), index=True)
    itemDescription = Column(VARCHAR(225), index=True)
    itemCost = Column(FLOAT, index=True)
    expectedItemQuantity = Column(INTEGER, index=True)
    receivedItemQuantity = Column(INTEGER, index=True)
    damagedItemQuantity = Column(INTEGER, index=True)
    containerId = Column(VARCHAR(225), index=True)

    shipmentHeader = relationship("ShipmentHeader", back_populates="shipmentDetails")
    receiptId = Column(VARCHAR(255), ForeignKey("shipmentHeader.receiptId"))

    invoiced = Column(BIT, index=True, default=0)  # Use BIT for Boolean in MSSQL

# PURCHASE ORDER
class PoHeader(Base):
    __tablename__ = "poHeader"
    poNumber = Column(VARCHAR(225), index=True, primary_key=True)
    shipByDate = Column(DATE, index=True)
    leadTime = Column(INTEGER, index=True)
    estimatedDeliveryDate = Column(DATE, index=True)
    totalQuantity = Column(INTEGER, index=True)
    totalCost = Column(FLOAT, index=True)
    totalTax = Column(FLOAT, index=True)
    currency = Column(VARCHAR(225), index=True)
    payment_term = Column(VARCHAR(225), index=True)
    poDetails = relationship("PoDetails", back_populates="poHeader")
    invoiceDetails = relationship("InvDetails", back_populates="poHeader")

    shipmentHeader = relationship("ShipmentHeader", back_populates="poHeader")

class PoDetails(Base):
    __tablename__ = 'poDetails'
    id = Column(INTEGER, primary_key=True, index=True, autoincrement=True)
    itemId = Column(VARCHAR(255), ForeignKey("itemMaster.itemId"), nullable=False, index=True)
    itemQuantity = Column(INTEGER, nullable=False)
    itemDescription = Column(VARCHAR(225), index=True)
    itemCost = Column(FLOAT, index=True)
    totalItemCost = Column(FLOAT, index=True)

    poId = Column(VARCHAR(225), ForeignKey("poHeader.poNumber"))
    poHeader = relationship("PoHeader", back_populates="poDetails")

    suppliers = relationship("Supplier", back_populates="poDetails")
    supplierId = Column(VARCHAR(225), ForeignKey("suppliers.supplierId"))
    itemMaster = relationship("ItemMaster", back_populates="itemPurchaseOrder")
    __table_args__ = (
        UniqueConstraint('poId', 'itemId', name='uq_po_item_unique'),
    )

# INVOICE
class InvHeader(Base):
    __tablename__ = "invoiceHeader"

    invoiceId = Column(VARCHAR(255), primary_key=True, index=True)
    invoicedate = Column(DATE, index=True)
    invoiceType = Column(VARCHAR(255), index=True)
    currency = Column(VARCHAR(255), index=True)
    payment_term = Column(VARCHAR(255), index=True)
    invoice_status = Column(VARCHAR(255), index=True)
    total_qty = Column(INTEGER, index=True)
    total_cost = Column(FLOAT, index=True)
    total_tax = Column(FLOAT, index=True)
    total_amount = Column(FLOAT, index=True)
    userInvNo = Column(VARCHAR(255), index=True)

    invoiceDetails = relationship("InvDetails", back_populates="invoiceHeader")
    stores = relationship("StoreDetails", secondary=invoice_store_association, back_populates="invoices")


class InvDetails(Base):
    __tablename__ = 'invoiceDetails'
    id = Column(INTEGER, primary_key=True, index=True, autoincrement=True)
    itemId = Column(VARCHAR(225), index=True)
    itemQuantity = Column(INTEGER, index=True)
    itemDescription = Column(VARCHAR(225), index=True)
    itemCost = Column(FLOAT, index=True)
    totalItemCost = Column(FLOAT, index=True)

    poId = Column(VARCHAR(225), ForeignKey("poHeader.poNumber"))
    invoiceNumber = Column(VARCHAR(255), ForeignKey("invoiceHeader.invoiceId"))

    invoiceHeader = relationship("InvHeader", back_populates="invoiceDetails")
    poHeader = relationship("PoHeader", back_populates="invoiceDetails")

# USER
class User(Base):
    __tablename__ = "users"

    id = Column(INTEGER, primary_key=True, index=True, autoincrement=True)
    name = Column(VARCHAR(16))
    email = Column(VARCHAR(16))
    nickname = Column(VARCHAR(16))
