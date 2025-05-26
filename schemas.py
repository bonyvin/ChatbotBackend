from typing import List, Optional
from pydantic import BaseModel
import datetime

#STORE
class StoreDetailsSchema(BaseModel):
    storeId :str
    storeName :str
    address :str
    city :str
    state :str
    zipCode :str
    phone :str

#PROMO
class PromotionHeaderSchema(BaseModel):
    promotionId :str
    componentId :str
    startDate:datetime.date
    endDate:datetime.date
    promotionType:str
    storeIds: List[str]
    

class PromotionDetailsSchema(BaseModel):
    promotionId  :str
    componentId  :str
    itemId :str
    discountType :str
    discountValue:float
#ITEM
class ItemMasterSchema(BaseModel):
    itemId :str
    itemDescription :str
    itemSecondaryDescription :str
    itemDepartment :str
    itemClass:str
    itemSubClass:str
    brand:str
    diffType1:int
    diffType2:int
    diffType3:int

class ItemSupplierSchema(BaseModel):
    id :int
    supplierCost:float
    supplierId :str
    itemId:str

class ItemDiffsSchema(BaseModel):
    id:int
    diffType:str
    diffId:str

#SHIPMENT
class ShipmentHeader(BaseModel):
    receiptId: str
    asnId : str
    expectedDate:datetime.date
    receivedDate:datetime.date
    receivedBy:str
    status:str
    sourceLocation:str
    destinationLocation:str
    totalQuantity:int
    totalCost:float
    poId : str

class ShipmentDetails(BaseModel):
    id: int
    itemId : str
    itemDescription:str
    itemCost:float
    expectedItemQuantity:int
    receivedItemQuantity:int
    damagedItemQuantity:int
    containerId: str
    receiptId:str
    invoiced:bool


#SUPPLIER
class SupplierCreate(BaseModel):
    supplierId: str 
    name: str
    email: str
    phone: str
    address: str
    lead_time:str

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatRequestUser(BaseModel):
    user_id:str

class UserSchema(BaseModel):
    id:int
    name:str
    email:str
    nickname:str

#PO
class poHeaderCreate(BaseModel):
    poNumber: str 
    shipByDate:  datetime.date
    leadTime: int
    estimatedDeliveryDate: datetime.date
    totalQuantity: int
    totalCost: float
    totalTax: float
    payment_term:str
    currency:str

class poDetailsCreate(BaseModel):
    # id:  int
    itemId :  str
    itemQuantity: int
    itemDescription: str
    itemCost: float
    totalItemCost: float
    poId : str
    supplierId:str

#INVOICE
class invHeaderCreate(BaseModel):
    invoiceId:str
    invoicedate:datetime.date
    # supplierId:str
    invoiceType:str
    currency:str
    payment_term:str
    invoice_status:str
    total_qty:int
    total_cost:float
    total_tax:float
    total_amount:float
    userInvNo:str
    # storeId:str
    storeIds: List[str]


class invDetailsCreate(BaseModel):
    # id:int
    itemId:str
    itemQuantity:int
    itemDescription:str
    itemCost:float
    totalItemCost:float
    poId:str
    invoiceNumber:str

class poDetailsSearch(BaseModel):
    poId:str

class invDetailsSerach(BaseModel):
    invId:str