from database import Base
from sqlalchemy import String, Column,Integer,ForeignKey,Date,Float,Boolean, Table
from sqlalchemy.orm import relationship

promotion_store_association = Table(
    "promotion_store_association",
    Base.metadata,
    Column("promotionId", String(255), ForeignKey("promotionHeader.promotionId"), primary_key=True),
    Column("storeId", String(255), ForeignKey("storeDetails.storeId"), primary_key=True)
)

# STORE
class StoreDetails(Base):
    __tablename__ = "storeDetails"
    storeId = Column(String(255), primary_key=True, index=True)
    storeName = Column(String(255), nullable=False)
    address = Column(String(255), nullable=False)
    city = Column(String(100), nullable=False)
    state = Column(String(100), nullable=False)
    zipCode = Column(String(20), nullable=False)
    phone = Column(String(20), nullable=False)
    
    invoiceHeader = relationship("InvHeader", back_populates="store", uselist=False)
    shipmentHeader = relationship("ShipmentHeader", back_populates="store", uselist=False)
    # Updated relationship: many-to-many with PromotionHeader via promotion_store table
    promotions = relationship("PromotionHeader", secondary=promotion_store_association, back_populates="stores")


# PROMOTION
class PromotionHeader(Base):
    __tablename__ = "promotionHeader"
    
    promotionId = Column(String(255), primary_key=True, index=True)
    componentId = Column(String(255), nullable=False, index=True, unique=True)
    startDate = Column(Date, index=True)
    endDate = Column(Date, index=True)
    promotionType = Column(String(20), nullable=False)
    
    # Removed single storeId field.
    # New relationship to allow multiple stores per promotion:
    stores = relationship("StoreDetails", secondary=promotion_store_association, back_populates="promotions")
    
    promotionDetailsbp = relationship(
        "PromotionDetails", 
        foreign_keys="[PromotionDetails.promotionId]",
        back_populates="promotionHeaders"
    )
# #STORE
# class StoreDetails(Base):
#     __tablename__ = "storeDetails"
#     storeId = Column(String(255), primary_key=True, index=True)
#     storeName = Column(String(255), nullable=False)
#     address = Column(String(255), nullable=False)
#     city = Column(String(100), nullable=False)
#     state = Column(String(100), nullable=False)
#     zipCode = Column(String(20), nullable=False)
#     phone = Column(String(20), nullable=False)
    
#     invoiceHeader = relationship("InvHeader", back_populates="store", uselist=False)
#     shipmentHeader = relationship("ShipmentHeader", back_populates="store", uselist=False)
#     promotionHeaders = relationship("PromotionHeader", back_populates="store")
    


# #PROMOTION
# class PromotionHeader(Base):
#     __tablename__ = "promotionHeader"
    
#     promotionId = Column(String(255), primary_key=True, index=True)
#     componentId = Column(String(255), nullable=False, index=True, unique=True)
#     startDate = Column(Date, index=True)
#     endDate = Column(Date, index=True)
#     promotionType = Column(String(20), nullable=False)
    
#     storeId = Column(String(255), ForeignKey("storeDetails.storeId"))
#     store = relationship("StoreDetails", back_populates="promotionHeaders")
#     promotionDetailsbp = relationship(
#         "PromotionDetails", 
#         foreign_keys="[PromotionDetails.promotionId]",  # Specify foreign key
#         back_populates="promotionHeaders"
#     )
# class PromotionHeader(Base):
#     __tablename__ = "promotionHeader"

#     promotionId = Column(String(255), primary_key=True, index=True)
#     componentId = Column(String(255), nullable=False, index=True, unique=True)
#     startDate = Column(Date, index=True)
#     endDate = Column(Date, index=True)
#     promotionType = Column(String(20), nullable=False)

#     promotionDetailsbp = relationship(
#         "PromotionDetails", 
#         foreign_keys="[PromotionDetails.promotionId]",  # Specify foreign key
#         back_populates="promotionHeaders"
#     )

class PromotionDetails(Base):
    __tablename__ = "promotionDetails"

    id = Column(Integer, primary_key=True, autoincrement=True)
    promotionId = Column(String(255), ForeignKey("promotionHeader.promotionId"))
    componentId = Column(String(255), ForeignKey("promotionHeader.componentId"))
    itemId = Column(String(255), ForeignKey("itemMaster.itemId"))
    discountType = Column(String(255), index=True)
    discountValue = Column(Float, nullable=False)

    promotionHeaders = relationship(
        "PromotionHeader", 
        foreign_keys=[promotionId],  # Specify foreign key
        back_populates="promotionDetailsbp"
    )
    itemMaster = relationship("ItemMaster", back_populates="itemPromotionDetail")

#ITEM
class ItemMaster(Base):
    __tablename__ = "itemMaster"

    itemId = Column(String(255), primary_key=True, index=True)
    itemDescription = Column(String(255), nullable=False)
    itemSecondaryDescription = Column(String(255), nullable=False)
    itemDepartment = Column(String(20), nullable=False)
    itemClass = Column(String(255), nullable=False)
    itemSubClass= Column(String(255), nullable=False)
    brand= Column(String(255), nullable=False)
    itemSuppliers = relationship("ItemSupplier", back_populates="itemMaster")
    diffType1=Column(Integer, ForeignKey("itemDiffs.id"))
    diffType2=Column(Integer, ForeignKey("itemDiffs.id"))
    diffType3=Column(Integer, ForeignKey("itemDiffs.id"))
    itemDiffs1 = relationship("ItemDiffs", foreign_keys=[diffType1])
    itemDiffs2 = relationship("ItemDiffs", foreign_keys=[diffType2])
    itemDiffs3 = relationship("ItemDiffs", foreign_keys=[diffType3])
    itemPromotionDetail = relationship("PromotionDetails", back_populates="itemMaster")

class ItemSupplier(Base):
    __tablename__ = "itemSupplier"

    id = Column(Integer, primary_key=True, index=True,autoincrement=True)
    supplierCost= Column(Float, nullable=False)
    itemMaster = relationship("ItemMaster", back_populates="itemSuppliers")
    supplier = relationship("Supplier", back_populates="itemSuppliers")
    supplierId = Column(String(225), ForeignKey("suppliers.supplierId"))
    itemId = Column(String(225), ForeignKey("itemMaster.itemId"))

class ItemDiffs(Base):
    __tablename__ = "itemDiffs"

    id = Column(Integer, primary_key=True, index=True,autoincrement=True)
    diffType= Column(String(255), nullable=False)
    diffId= Column(String(255), nullable=False)
    # itemMaster=relationship("ItemMaster", back_populates="itemDiffs")

#SUPPLIER
class Supplier(Base):
    __tablename__ = "suppliers"

    supplierId = Column(String(255), primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    phone = Column(String(20), nullable=False)
    address = Column(String(255), nullable=False)
    lead_time= Column(String(255), nullable=False)

    poDetails=relationship("PoDetails", back_populates="suppliers")
    itemSuppliers = relationship("ItemSupplier", back_populates="supplier")
    
#SHIPMENT
class ShipmentHeader(Base):
    __tablename__ = "shipmentHeader"
    
    receiptId = Column(String(255), primary_key=True, index=True)
    asnId = Column(String(225), index=True)
    expectedDate = Column(Date, index=True)
    receivedDate = Column(Date, index=True)
    receivedBy = Column(String(225), index=True)
    status = Column(String(225), index=True)
    sourceLocation = Column(String(225), index=True)
    destinationLocation =Column(String(255), ForeignKey("storeDetails.storeId"), unique=True)
    totalQuantity = Column(Integer, index=True)
    totalCost = Column(Float, index=True)
    poHeader= relationship("PoHeader", back_populates="shipmentHeader")

    poId = Column(String(225), ForeignKey("poHeader.poNumber"))
    store = relationship("StoreDetails", back_populates="shipmentHeader")
    shipmentDetails = relationship("ShipmentDetails", back_populates="shipmentHeader")

# class ShipmentHeader(Base):
#     __tablename__='shipmentHeader'
#     receiptId= Column(String(255),primary_key=True, index=True)
#     asnId = Column(String(225), index=True )
#     expectedDate=Column(Date , index=True)
#     receivedDate=Column(Date , index=True)
#     receivedBy=Column(String(225), index=True )
#     status=Column(String(225), index=True )

#     sourceLocation=Column(String(225), index=True )
#     destinationLocation=Column(String(225), index=True )
#     totalQuantity=Column(Integer, index=True )
#     totalCost=Column(Float, index=True )

#     poHeader= relationship("PoHeader", back_populates="shipmentHeader")
#     poId = Column(String(225), ForeignKey("poHeader.poNumber"))

#     shipmentDetails = relationship("ShipmentDetails", back_populates="shipmentHeader")


class ShipmentDetails(Base):
    __tablename__='shipmentDetails'
    id= Column(Integer,primary_key=True, index=True,autoincrement=True)
    itemId = Column(String(225), index=True )
    itemDescription=Column(String(225), index=True )
    itemCost=Column(Float, index=True )
    expectedItemQuantity=Column(Integer, index=True )
    receivedItemQuantity=Column(Integer, index=True )
    damagedItemQuantity=Column(Integer, index=True )
    containerId= Column(String(225), index=True )

    shipmentHeader= relationship("ShipmentHeader", back_populates="shipmentDetails")
    receiptId=Column(String(255), ForeignKey("shipmentHeader.receiptId"))

    invoiced = Column(Boolean, index=True, default=False) 
    
#PURCHASE ORDER
class PoHeader(Base):
    __tablename__ = "poHeader"
    # id = Column(Integer, index=True, autoincrement=True)
    poNumber=Column(String(225) , index=True, primary_key=True)
    shipByDate=Column(Date , index=True)
    leadTime=Column(Integer , index=True)
    estimatedDeliveryDate=Column(Date , index=True)
    totalQuantity=Column(Integer , index=True)
    totalCost=Column(Float , index=True)
    totalTax=Column(Float , index=True)
    currency=Column(String(225),index=True)
    payment_term=Column(String(225),index=True)
    poDetails = relationship("PoDetails", back_populates="poHeader")
    invoiceDetails = relationship("InvDetails", back_populates="poHeader")

    shipmentHeader = relationship("ShipmentHeader", back_populates="poHeader")

class PoDetails(Base):
    __tablename__='poDetails'
    id= Column(Integer,primary_key=True, index=True,autoincrement=True)
    itemId = Column(String(225), index=True )
    itemQuantity=Column(Integer, index=True )
    # supplierId = Column(String(225),index=True)
    itemDescription=Column(String(225), index=True )
    itemCost=Column(Float, index=True )
    totalItemCost=Column(Float, index=True )

    poId = Column(String(225), ForeignKey("poHeader.poNumber"))
    poHeader= relationship("PoHeader", back_populates="poDetails")

    suppliers=relationship("Supplier", back_populates="poDetails")
    supplierId = Column(String(225), ForeignKey("suppliers.supplierId"))

#INVOICE
class InvHeader(Base):
    __tablename__ = "invoiceHeader"
    
    invoiceId = Column(String(255), primary_key=True, index=True)
    invoicedate = Column(Date, index=True)
    invoiceType = Column(String(255), index=True)
    currency = Column(String(255), index=True)
    payment_term = Column(String(255), index=True)
    invoice_status = Column(String(255), index=True)
    total_qty = Column(Integer, index=True)
    total_cost = Column(Float, index=True)
    total_tax = Column(Float, index=True)
    total_amount = Column(Float, index=True)
    userInvNo = Column(String(255), index=True)
    
    invoiceDetails = relationship("InvDetails", back_populates="invoiceHeader")
    storeId = Column(String(255), ForeignKey("storeDetails.storeId"), unique=True)
    store = relationship("StoreDetails", back_populates="invoiceHeader")
# class InvHeader(Base):
#     __tablename__='invoiceHeader'
#     invoiceId = Column(String(225),primary_key=True,index=True)
#     invoicedate = Column(Date , index=True)
#     invoiceType = Column(String(225),index=True)
#     currency=Column(String(225),index=True)
#     payment_term=Column(String(225),index=True)
#     invoice_status = Column(String(225),index=True)
#     total_qty=Column(Integer,index=True)
#     total_cost=Column(Float,index=True)
#     total_tax=Column(Float,index=True)
#     total_amount= Column(Float,index=True)
#     userInvNo = Column(String(225),index=True)
#     invoiceDetails = relationship("InvDetails", back_populates="invoiceHeader")


class InvDetails(Base):
    __tablename__='invoiceDetails'
    id= Column(Integer,primary_key=True, index=True,autoincrement=True)
    itemId = Column(String(225), index=True )
    itemQuantity=Column(Integer, index=True )
    itemDescription=Column(String(225), index=True )
    itemCost=Column(Float, index=True )
    totalItemCost=Column(Float, index=True )

    poId = Column(String(225), ForeignKey("poHeader.poNumber"))
    invoiceNumber = Column(String(225), ForeignKey("invoiceHeader.invoiceId"))

    invoiceHeader= relationship("InvHeader", back_populates="invoiceDetails")
    poHeader= relationship("PoHeader", back_populates="invoiceDetails")


#USER
class User(Base):
    __tablename__="users"

    id = Column(Integer,primary_key=True,index=True,autoincrement=True)
    name=Column(String(16))
    email=Column(String(16))
    nickname=Column(String(16))