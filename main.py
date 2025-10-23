import os
import uvicorn
import re
from datetime import datetime, timedelta
from typing import List, Optional, Annotated
from enum import Enum

from fastapi import Depends, FastAPI, HTTPException, status, Request, Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field, EmailStr, ConfigDict, field_validator
from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer,
                        String, create_engine, Enum as SQLAlchemyEnum, func, CheckConstraint)
from sqlalchemy.orm import relationship, sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./erp_full_v4.db")
SECRET_KEY = os.getenv("SECRET_KEY")
REFRESH_SECRET_KEY = os.getenv("REFRESH_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7
CLIENT_ORIGINS = os.getenv("CLIENT_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
PASSWORD_REGEX = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
limiter = Limiter(key_func=get_remote_address)

class Role(str, Enum):
    admin = "admin"
    hr_manager = "hr_manager"
    sales_rep = "sales_rep"
    inventory_manager = "inventory_manager"
    finance_manager = "finance_manager"
    viewer = "viewer"

class InvoiceStatus(str, Enum):
    draft = "draft"
    sent = "sent"
    paid = "paid"
    partially_paid = "partially_paid"
    overdue = "overdue"
    cancelled = "cancelled"

class PaymentMethod(str, Enum):
    credit_card = "credit_card"
    bank_transfer = "bank_transfer"
    cash = "cash"
    other = "other"

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class AccessToken(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    username: Optional[str] = None
    token_type: str = "access"

class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None
    is_active: Optional[bool] = True

class UserCreate(UserBase):
    password: str = Field(
        ...,
        min_length=8,
        description="Password must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, one number, and one special character."
    )
    role: Role = Role.viewer

    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        if not re.fullmatch(PASSWORD_REGEX, v):
            raise ValueError(
                'Password must be at least 8 characters long and contain at least '
                'one uppercase letter, one lowercase letter, one number, and one special character.'
            )
        return v

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None
    role: Optional[Role] = None

class User(UserBase):
    id: int
    role: Role
    model_config = ConfigDict(from_attributes=True)

class EmployeeBase(BaseModel):
    user_id: int
    department: str
    position: str
    start_date: datetime
    salary: float = Field(..., gt=0)

class EmployeeCreate(EmployeeBase):
    pass

class EmployeeUpdate(BaseModel):
    department: Optional[str] = None
    position: Optional[str] = None
    salary: Optional[float] = Field(None, gt=0)

class Employee(EmployeeBase):
    id: int
    user: User
    model_config = ConfigDict(from_attributes=True)

class ProductBase(BaseModel):
    name: str
    sku: str = Field(..., min_length=1)
    description: Optional[str] = None
    price: float = Field(..., gt=0)
    quantity_in_stock: int = Field(..., ge=0)

class ProductCreate(ProductBase):
    pass

class ProductUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = Field(None, gt=0)
    quantity_in_stock: Optional[int] = Field(None, ge=0)

class Product(ProductBase):
    id: int
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)

class CustomerBase(BaseModel):
    company_name: str
    contact_name: Optional[str] = None
    email: EmailStr
    phone: Optional[str] = None
    address: Optional[str] = None

class CustomerCreate(CustomerBase):
    pass

class CustomerUpdate(BaseModel):
    company_name: Optional[str] = None
    contact_name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    address: Optional[str] = None

class Customer(CustomerBase):
    id: int
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class SupplierBase(BaseModel):
    name: str
    contact_info: str

class SupplierCreate(SupplierBase):
    pass

class SupplierUpdate(BaseModel):
    name: Optional[str] = None
    contact_info: Optional[str] = None

class Supplier(SupplierBase):
    id: int
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class SalesOrderItemBase(BaseModel):
    product_id: int
    quantity: int = Field(..., gt=0)
    unit_price: float = Field(..., gt=0)

class SalesOrderItemCreate(SalesOrderItemBase):
    pass

class SalesOrderItem(SalesOrderItemBase):
    id: int
    sales_order_id: int
    product: Product
    model_config = ConfigDict(from_attributes=True)

class SalesOrderBase(BaseModel):
    customer_id: int
    order_date: datetime = Field(default_factory=datetime.utcnow)
    status: str = "pending"
    shipping_address: Optional[str] = None

class SalesOrderCreate(SalesOrderBase):
    items: List[SalesOrderItemCreate]

class SalesOrderUpdate(BaseModel):
    status: Optional[str] = None
    shipping_address: Optional[str] = None

class SalesOrder(SalesOrderBase):
    id: int
    total_amount: float
    customer: Customer
    items: List[SalesOrderItem]
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class PurchaseOrderItemBase(BaseModel):
    product_id: int
    quantity: int = Field(..., gt=0)
    unit_price: float = Field(..., gt=0)

class PurchaseOrderItemCreate(PurchaseOrderItemBase):
    pass

class PurchaseOrderItem(PurchaseOrderItemBase):
    id: int
    purchase_order_id: int
    product: Product
    model_config = ConfigDict(from_attributes=True)

class PurchaseOrderBase(BaseModel):
    supplier_id: int
    order_date: datetime = Field(default_factory=datetime.utcnow)
    status: str = "draft"
    expected_delivery_date: Optional[datetime] = None

class PurchaseOrderCreate(PurchaseOrderBase):
    items: List[PurchaseOrderItemCreate]

class PurchaseOrderUpdate(BaseModel):
    status: Optional[str] = None
    expected_delivery_date: Optional[datetime] = None

class PurchaseOrder(PurchaseOrderBase):
    id: int
    total_amount: float
    supplier: Supplier
    items: List[PurchaseOrderItem]
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class PaymentBase(BaseModel):
    invoice_id: int
    amount_paid: float = Field(..., gt=0)
    payment_method: PaymentMethod
    transaction_id: Optional[str] = None

class PaymentCreate(PaymentBase):
    payment_date: datetime = Field(default_factory=datetime.utcnow)

class Payment(PaymentBase):
    id: int
    payment_date: datetime
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class InvoiceBase(BaseModel):
    customer_id: Optional[int] = None
    supplier_id: Optional[int] = None
    sales_order_id: Optional[int] = None
    purchase_order_id: Optional[int] = None
    issue_date: datetime = Field(default_factory=datetime.utcnow)
    due_date: datetime
    total_amount: float = Field(..., gt=0)
    status: InvoiceStatus = InvoiceStatus.draft

class InvoiceCreate(BaseModel):
    due_date: datetime
    sales_order_id: Optional[int] = None
    purchase_order_id: Optional[int] = None
    model_config = ConfigDict(extra="ignore")

class InvoiceUpdate(BaseModel):
    due_date: Optional[datetime] = None
    status: Optional[InvoiceStatus] = None

class Invoice(InvoiceBase):
    id: int
    created_at: datetime
    payments: List[Payment]
    customer: Optional[Customer] = None
    supplier: Optional[Supplier] = None
    total_paid: float = 0.0
    balance_due: float = 0.0
    model_config = ConfigDict(from_attributes=True)

class AuditLog(BaseModel):
    id: int
    timestamp: datetime
    user_email: str
    action: str
    model_config = ConfigDict(from_attributes=True)

class UserModel(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    role = Column(SQLAlchemyEnum(Role), default=Role.viewer, nullable=False)
    employee = relationship("EmployeeModel", back_populates="user", uselist=False, cascade="all, delete-orphan")

class EmployeeModel(Base):
    __tablename__ = "employees"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    department = Column(String)
    position = Column(String)
    start_date = Column(DateTime)
    salary = Column(Float, CheckConstraint('salary > 0'))
    user = relationship("UserModel", back_populates="employee")

class ProductModel(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    sku = Column(String, unique=True, index=True, nullable=False)
    description = Column(String)
    price = Column(Float, CheckConstraint('price > 0'), nullable=False)
    quantity_in_stock = Column(Integer, CheckConstraint('quantity_in_stock >= 0'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class CustomerModel(Base):
    __tablename__ = "customers"
    id = Column(Integer, primary_key=True, index=True)
    company_name = Column(String, index=True, nullable=False)
    contact_name = Column(String)
    email = Column(String, unique=True, index=True, nullable=False)
    phone = Column(String)
    address = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    sales_orders = relationship("SalesOrderModel", back_populates="customer")
    invoices = relationship("InvoiceModel", back_populates="customer")

class SupplierModel(Base):
    __tablename__ = "suppliers"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    contact_info = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    purchase_orders = relationship("PurchaseOrderModel", back_populates="supplier")
    invoices = relationship("InvoiceModel", back_populates="supplier")

class SalesOrderModel(Base):
    __tablename__ = "sales_orders"
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    order_date = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="pending")
    total_amount = Column(Float, nullable=False)
    shipping_address = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    customer = relationship("CustomerModel", back_populates="sales_orders")
    items = relationship("SalesOrderItemModel", back_populates="order", cascade="all, delete-orphan")
    invoice = relationship("InvoiceModel", back_populates="sales_order", uselist=False)

class SalesOrderItemModel(Base):
    __tablename__ = "sales_order_items"
    id = Column(Integer, primary_key=True, index=True)
    sales_order_id = Column(Integer, ForeignKey("sales_orders.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    quantity = Column(Integer, CheckConstraint('quantity > 0'), nullable=False)
    unit_price = Column(Float, CheckConstraint('unit_price > 0'), nullable=False)
    order = relationship("SalesOrderModel", back_populates="items")
    product = relationship("ProductModel")

class PurchaseOrderModel(Base):
    __tablename__ = "purchase_orders"
    id = Column(Integer, primary_key=True, index=True)
    supplier_id = Column(Integer, ForeignKey("suppliers.id"), nullable=False)
    order_date = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="draft")
    total_amount = Column(Float, nullable=False)
    expected_delivery_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    supplier = relationship("SupplierModel", back_populates="purchase_orders")
    items = relationship("PurchaseOrderItemModel", back_populates="order", cascade="all, delete-orphan")
    invoice = relationship("InvoiceModel", back_populates="purchase_order", uselist=False)

class PurchaseOrderItemModel(Base):
    __tablename__ = "purchase_order_items"
    id = Column(Integer, primary_key=True, index=True)
    purchase_order_id = Column(Integer, ForeignKey("purchase_orders.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    quantity = Column(Integer, CheckConstraint('quantity > 0'), nullable=False)
    unit_price = Column(Float, CheckConstraint('unit_price > 0'), nullable=False)
    order = relationship("PurchaseOrderModel", back_populates="items")
    product = relationship("ProductModel")

class InvoiceModel(Base):
    __tablename__ = "invoices"
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=True)
    supplier_id = Column(Integer, ForeignKey("suppliers.id"), nullable=True)
    sales_order_id = Column(Integer, ForeignKey("sales_orders.id"), unique=True, nullable=True)
    purchase_order_id = Column(Integer, ForeignKey("purchase_orders.id"), unique=True, nullable=True)
    issue_date = Column(DateTime, default=datetime.utcnow)
    due_date = Column(DateTime, nullable=False)
    total_amount = Column(Float, CheckConstraint('total_amount > 0'), nullable=False)
    status = Column(SQLAlchemyEnum(InvoiceStatus), default=InvoiceStatus.draft, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    customer = relationship("CustomerModel", back_populates="invoices")
    supplier = relationship("SupplierModel", back_populates="invoices")
    sales_order = relationship("SalesOrderModel", back_populates="invoice")
    purchase_order = relationship("PurchaseOrderModel", back_populates="invoice")
    payments = relationship("PaymentModel", back_populates="invoice", cascade="all, delete-orphan")

class PaymentModel(Base):
    __tablename__ = "payments"
    id = Column(Integer, primary_key=True, index=True)
    invoice_id = Column(Integer, ForeignKey("invoices.id"), nullable=False)
    payment_date = Column(DateTime, default=datetime.utcnow)
    amount_paid = Column(Float, CheckConstraint('amount_paid > 0'), nullable=False)
    payment_method = Column(SQLAlchemyEnum(PaymentMethod), nullable=False)
    transaction_id = Column(String, unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    invoice = relationship("InvoiceModel", back_populates="payments")

class AuditLogModel(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_email = Column(String, index=True, nullable=False)
    action = Column(String, nullable=False)

def create_db_and_tables():
    Base.metadata.create_all(bind=engine)

app = FastAPI(title="Full ERP API (v4)", version="4.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CLIENT_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # This is the corrected CSP header.
    # We've added 'unsafe-inline' to script-src to allow the Swagger UI initialization script to run.
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'"
    return response

@app.on_event("startup")
def on_startup():
    if not SECRET_KEY or not REFRESH_SECRET_KEY:
        raise ValueError("Missing critical environment variables: SECRET_KEY, REFRESH_SECRET_KEY")
    create_db_and_tables()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_audit_log(db: Session, user_email: str, action: str):
    try:
        db_log = AuditLogModel(user_email=user_email, action=action)
        db.add(db_log)
        db.commit()
    except Exception:
        db.rollback()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_jwt_token(data: dict, expires_delta: timedelta, secret_key: str):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=ALGORITHM)
    return encoded_jwt

def decode_jwt_token(token: str, secret_key: str, credentials_exception: HTTPException):
    try:
        payload = jwt.decode(token, secret_key, algorithms=[ALGORITHM])
        if payload.get("sub") is None or payload.get("type") is None:
            raise credentials_exception
        return payload
    except JWTError:
        raise credentials_exception

def get_user_by_email(db: Session, email: str):
    return db.query(UserModel).filter(UserModel.email == email).first()

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)], db: Session = Depends(get_db)):
    credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    payload = decode_jwt_token(token, SECRET_KEY, credentials_exception)
    if payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type, expected 'access'", headers={"WWW-Authenticate": "Bearer"})
    user = get_user_by_email(db, email=payload.get("sub"))
    if user is None:
        raise credentials_exception
    return user

async def get_current_user_from_refresh_token(token: Annotated[str, Depends(oauth2_scheme)], db: Session = Depends(get_db)):
    credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate refresh token", headers={"WWW-Authenticate": "Bearer"})
    payload = decode_jwt_token(token, REFRESH_SECRET_KEY, credentials_exception)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type, expected 'refresh'", headers={"WWW-Authenticate": "Bearer"})
    user = get_user_by_email(db, email=payload.get("sub"))
    if user is None or not user.is_active:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: Annotated[User, Depends(get_current_user)]):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def require_role(required_roles: List[Role]):
    def role_checker(current_user: User = Depends(get_current_active_user)):
        if current_user.role not in required_roles and current_user.role != Role.admin:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="The user does not have sufficient privileges for this resource.")
        return current_user
    return role_checker

admin_only = require_role([Role.admin])
hr_access = require_role([Role.hr_manager])
sales_access = require_role([Role.sales_rep])
inventory_access = require_role([Role.inventory_manager])
finance_access = require_role([Role.finance_manager])
viewer_access = require_role([Role.viewer, Role.sales_rep, Role.inventory_manager, Role.hr_manager, Role.finance_manager])

@app.post("/token", response_model=Token, tags=["Authentication"])
@limiter.limit("5/minute")
async def login_for_access_token(request: Request, form_data: Annotated[OAuth2PasswordRequestForm, Depends()], db: Session = Depends(get_db)):
    user = get_user_by_email(db, form_data.username)
    if not user or not user.is_active or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    refresh_token_expires = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    access_token = create_jwt_token(data={"sub": user.email, "type": "access"}, expires_delta=access_token_expires, secret_key=SECRET_KEY)
    refresh_token = create_jwt_token(data={"sub": user.email, "type": "refresh"}, expires_delta=refresh_token_expires, secret_key=REFRESH_SECRET_KEY)
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}

@app.post("/token/refresh", response_model=AccessToken, tags=["Authentication"])
async def refresh_access_token(current_user: Annotated[User, Depends(get_current_user_from_refresh_token)]):
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    new_access_token = create_jwt_token(data={"sub": current_user.email, "type": "access"}, expires_delta=access_token_expires, secret_key=SECRET_KEY)
    return {"access_token": new_access_token, "token_type": "bearer"}

@app.get("/users/me/", response_model=User, tags=["Users"])
async def read_users_me(current_user: Annotated[User, Depends(get_current_active_user)]):
    return current_user

@app.post("/users/", response_model=User, status_code=201, tags=["Users"])
def create_user(user: UserCreate, db: Session = Depends(get_db), current_user: User = Depends(admin_only)):
    if get_user_by_email(db, user.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = get_password_hash(user.password)
    db_user = UserModel(email=user.email, hashed_password=hashed_password, full_name=user.full_name, role=user.role, is_active=user.is_active)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    create_audit_log(db, current_user.email, f"Created user {user.email} with role {user.role.value}")
    return db_user

@app.get("/users/", response_model=List[User], tags=["Users"])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), current_user: User = Depends(admin_only)):
    return db.query(UserModel).offset(skip).limit(limit).all()

@app.get("/users/{user_id}", response_model=User, tags=["Users"])
def read_user(user_id: int, db: Session = Depends(get_db), current_user: User = Depends(admin_only)):
    db_user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@app.put("/users/{user_id}", response_model=User, tags=["Users"])
def update_user(user_id: int, user_update: UserUpdate, db: Session = Depends(get_db), current_user: User = Depends(admin_only)):
    db_user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    if user_update.email and get_user_by_email(db, user_update.email) and db_user.email != user_update.email:
        raise HTTPException(status_code=400, detail="Email already registered")
    update_data = user_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_user, key, value)
    db.commit()
    db.refresh(db_user)
    create_audit_log(db, current_user.email, f"Updated user ID {user_id}")
    return db_user

@app.delete("/users/{user_id}", status_code=204, tags=["Users"])
def delete_user(user_id: int, db: Session = Depends(get_db), current_user: User = Depends(admin_only)):
    db_user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    if db_user.id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own user account")
    user_email = db_user.email
    db.delete(db_user)
    db.commit()
    create_audit_log(db, current_user.email, f"Deleted user {user_email} (ID: {user_id})")
    return

@app.post("/employees/", response_model=Employee, status_code=201, tags=["HR"])
def create_employee(employee: EmployeeCreate, db: Session = Depends(get_db), current_user: User = Depends(hr_access)):
    db_user = db.query(UserModel).filter(UserModel.id == employee.user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    if db.query(EmployeeModel).filter(EmployeeModel.user_id == employee.user_id).first():
        raise HTTPException(status_code=400, detail="Employee record already exists for this user")
    db_employee = EmployeeModel(**employee.model_dump())
    db.add(db_employee)
    db.commit()
    db.refresh(db_employee)
    create_audit_log(db, current_user.email, f"Created employee record for user {db_user.email}")
    return db_employee

@app.get("/employees/", response_model=List[Employee], tags=["HR"])
def read_employees(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), current_user: User = Depends(hr_access)):
    return db.query(EmployeeModel).offset(skip).limit(limit).all()

@app.get("/employees/{employee_id}", response_model=Employee, tags=["HR"])
def read_employee(employee_id: int, db: Session = Depends(get_db), current_user: User = Depends(hr_access)):
    db_employee = db.query(EmployeeModel).filter(EmployeeModel.id == employee_id).first()
    if not db_employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    return db_employee

@app.put("/employees/{employee_id}", response_model=Employee, tags=["HR"])
def update_employee(employee_id: int, employee_update: EmployeeUpdate, db: Session = Depends(get_db), current_user: User = Depends(hr_access)):
    db_employee = db.query(EmployeeModel).filter(EmployeeModel.id == employee_id).first()
    if not db_employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    update_data = employee_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_employee, key, value)
    db.commit()
    db.refresh(db_employee)
    create_audit_log(db, current_user.email, f"Updated employee record ID {employee_id}")
    return db_employee

@app.delete("/employees/{employee_id}", status_code=204, tags=["HR"])
def delete_employee(employee_id: int, db: Session = Depends(get_db), current_user: User = Depends(hr_access)):
    db_employee = db.query(EmployeeModel).filter(EmployeeModel.id == employee_id).first()
    if not db_employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    db.delete(db_employee)
    db.commit()
    create_audit_log(db, current_user.email, f"Deleted employee record ID {employee_id}")
    return

@app.post("/products/", response_model=Product, status_code=201, tags=["Inventory"])
def create_product(product: ProductCreate, db: Session = Depends(get_db), current_user: User = Depends(inventory_access)):
    if db.query(ProductModel).filter(ProductModel.sku == product.sku).first():
        raise HTTPException(status_code=400, detail=f"Product with SKU '{product.sku}' already exists")
    db_product = ProductModel(**product.model_dump())
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    create_audit_log(db, current_user.email, f"Created product '{product.name}' with SKU '{product.sku}'")
    return db_product

@app.get("/products/", response_model=List[Product], tags=["Inventory"])
def read_products(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), current_user: User = Depends(viewer_access)):
    return db.query(ProductModel).offset(skip).limit(limit).all()

@app.get("/products/{product_id}", response_model=Product, tags=["Inventory"])
def read_product(product_id: int, db: Session = Depends(get_db), current_user: User = Depends(viewer_access)):
    db_product = db.query(ProductModel).filter(ProductModel.id == product_id).first()
    if not db_product:
        raise HTTPException(status_code=404, detail="Product not found")
    return db_product

@app.put("/products/{product_id}", response_model=Product, tags=["Inventory"])
def update_product(product_id: int, product_update: ProductUpdate, db: Session = Depends(get_db), current_user: User = Depends(inventory_access)):
    db_product = db.query(ProductModel).filter(ProductModel.id == product_id).first()
    if not db_product:
        raise HTTPException(status_code=404, detail="Product not found")
    update_data = product_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_product, key, value)
    db.commit()
    db.refresh(db_product)
    create_audit_log(db, current_user.email, f"Updated product '{db_product.name}' (ID: {product_id})")
    return db_product

@app.delete("/products/{product_id}", status_code=204, tags=["Inventory"])
def delete_product(product_id: int, db: Session = Depends(get_db), current_user: User = Depends(inventory_access)):
    db_product = db.query(ProductModel).filter(ProductModel.id == product_id).first()
    if not db_product:
        raise HTTPException(status_code=404, detail="Product not found")
    product_name = db_product.name
    db.delete(db_product)
    db.commit()
    create_audit_log(db, current_user.email, f"Deleted product '{product_name}' (ID: {product_id})")
    return

@app.post("/customers/", response_model=Customer, status_code=201, tags=["Sales"])
def create_customer(customer: CustomerCreate, db: Session = Depends(get_db), current_user: User = Depends(sales_access)):
    if db.query(CustomerModel).filter(CustomerModel.email == customer.email).first():
        raise HTTPException(status_code=400, detail="Customer with this email already exists")
    db_customer = CustomerModel(**customer.model_dump())
    db.add(db_customer)
    db.commit()
    db.refresh(db_customer)
    create_audit_log(db, current_user.email, f"Created customer '{customer.company_name}'")
    return db_customer

@app.get("/customers/", response_model=List[Customer], tags=["Sales"])
def read_customers(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), current_user: User = Depends(sales_access)):
    return db.query(CustomerModel).offset(skip).limit(limit).all()

@app.get("/customers/{customer_id}", response_model=Customer, tags=["Sales"])
def read_customer(customer_id: int, db: Session = Depends(get_db), current_user: User = Depends(sales_access)):
    db_customer = db.query(CustomerModel).filter(CustomerModel.id == customer_id).first()
    if not db_customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return db_customer

@app.put("/customers/{customer_id}", response_model=Customer, tags=["Sales"])
def update_customer(customer_id: int, customer_update: CustomerUpdate, db: Session = Depends(get_db), current_user: User = Depends(sales_access)):
    db_customer = db.query(CustomerModel).filter(CustomerModel.id == customer_id).first()
    if not db_customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    if customer_update.email and db.query(CustomerModel).filter(CustomerModel.email == customer_update.email).first() and db_customer.email != customer_update.email:
        raise HTTPException(status_code=400, detail="Customer with this email already exists")
    update_data = customer_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_customer, key, value)
    db.commit()
    db.refresh(db_customer)
    create_audit_log(db, current_user.email, f"Updated customer '{db_customer.company_name}' (ID: {customer_id})")
    return db_customer

@app.delete("/customers/{customer_id}", status_code=204, tags=["Sales"])
def delete_customer(customer_id: int, db: Session = Depends(get_db), current_user: User = Depends(sales_access)):
    db_customer = db.query(CustomerModel).filter(CustomerModel.id == customer_id).first()
    if not db_customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    customer_name = db_customer.company_name
    db.delete(db_customer)
    db.commit()
    create_audit_log(db, current_user.email, f"Deleted customer '{customer_name}' (ID: {customer_id})")
    return

@app.post("/sales_orders/", response_model=SalesOrder, status_code=201, tags=["Sales"])
def create_sales_order(order: SalesOrderCreate, db: Session = Depends(get_db), current_user: User = Depends(sales_access)):
    customer = db.query(CustomerModel).filter(CustomerModel.id == order.customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    try:
        with db.begin_nested():
            product_ids = [item.product_id for item in order.items]
            products = db.query(ProductModel).filter(ProductModel.id.in_(product_ids)).with_for_update().all()
            product_map = {p.id: p for p in products}
            total_amount = 0
            order_items = []
            for item_data in order.items:
                product = product_map.get(item_data.product_id)
                if not product:
                    raise HTTPException(status_code=404, detail=f"Product with id {item_data.product_id} not found")
                if product.quantity_in_stock < item_data.quantity:
                    raise HTTPException(status_code=400, detail=f"Not enough stock for {product.name}. Available: {product.quantity_in_stock}")
                total_amount += item_data.quantity * item_data.unit_price
                product.quantity_in_stock -= item_data.quantity
                order_items.append(SalesOrderItemModel(**item_data.model_dump()))
            db_order = SalesOrderModel(customer_id=order.customer_id, order_date=order.order_date, status=order.status, shipping_address=order.shipping_address or customer.address, total_amount=total_amount, items=order_items)
            db.add(db_order)
            db.flush()
        db.commit()
        db.refresh(db_order)
        create_audit_log(db, current_user.email, f"Created sales order ID {db_order.id} for {customer.company_name}")
        return db_order
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sales_orders/", response_model=List[SalesOrder], tags=["Sales"])
def read_sales_orders(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), current_user: User = Depends(sales_access)):
    return db.query(SalesOrderModel).offset(skip).limit(limit).all()

@app.get("/sales_orders/{order_id}", response_model=SalesOrder, tags=["Sales"])
def read_sales_order(order_id: int, db: Session = Depends(get_db), current_user: User = Depends(sales_access)):
    db_order = db.query(SalesOrderModel).filter(SalesOrderModel.id == order_id).first()
    if not db_order:
        raise HTTPException(status_code=404, detail="Sales order not found")
    return db_order

@app.put("/sales_orders/{order_id}", response_model=SalesOrder, tags=["Sales"])
def update_sales_order_status(order_id: int, order_update: SalesOrderUpdate, db: Session = Depends(get_db), current_user: User = Depends(sales_access)):
    db_order = db.query(SalesOrderModel).filter(SalesOrderModel.id == order_id).first()
    if not db_order:
        raise HTTPException(status_code=404, detail="Sales order not found")
    update_data = order_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_order, key, value)
    db.commit()
    db.refresh(db_order)
    create_audit_log(db, current_user.email, f"Updated sales order ID {order_id} status to '{db_order.status}'")
    return db_order

@app.post("/suppliers/", response_model=Supplier, status_code=201, tags=["Purchasing"])
def create_supplier(supplier: SupplierCreate, db: Session = Depends(get_db), current_user: User = Depends(inventory_access)):
    if db.query(SupplierModel).filter(SupplierModel.name == supplier.name).first():
        raise HTTPException(status_code=400, detail="Supplier with this name already exists")
    db_supplier = SupplierModel(**supplier.model_dump())
    db.add(db_supplier)
    db.commit()
    db.refresh(db_supplier)
    create_audit_log(db, current_user.email, f"Created supplier '{supplier.name}'")
    return db_supplier

@app.get("/suppliers/", response_model=List[Supplier], tags=["Purchasing"])
def read_suppliers(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), current_user: User = Depends(inventory_access)):
    return db.query(SupplierModel).offset(skip).limit(limit).all()

@app.get("/suppliers/{supplier_id}", response_model=Supplier, tags=["Purchasing"])
def read_supplier(supplier_id: int, db: Session = Depends(get_db), current_user: User = Depends(inventory_access)):
    db_supplier = db.query(SupplierModel).filter(SupplierModel.id == supplier_id).first()
    if not db_supplier:
        raise HTTPException(status_code=404, detail="Supplier not found")
    return db_supplier

@app.put("/suppliers/{supplier_id}", response_model=Supplier, tags=["Purchasing"])
def update_supplier(supplier_id: int, supplier_update: SupplierUpdate, db: Session = Depends(get_db), current_user: User = Depends(inventory_access)):
    db_supplier = db.query(SupplierModel).filter(SupplierModel.id == supplier_id).first()
    if not db_supplier:
        raise HTTPException(status_code=404, detail="Supplier not found")
    update_data = supplier_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_supplier, key, value)
    db.commit()
    db.refresh(db_supplier)
    create_audit_log(db, current_user.email, f"Updated supplier '{db_supplier.name}' (ID: {supplier_id})")
    return db_supplier

@app.delete("/suppliers/{supplier_id}", status_code=204, tags=["Purchasing"])
def delete_supplier(supplier_id: int, db: Session = Depends(get_db), current_user: User = Depends(inventory_access)):
    db_supplier = db.query(SupplierModel).filter(SupplierModel.id == supplier_id).first()
    if not db_supplier:
        raise HTTPException(status_code=404, detail="Supplier not found")
    supplier_name = db_supplier.name
    db.delete(db_supplier)
    db.commit()
    create_audit_log(db, current_user.email, f"Deleted supplier '{supplier_name}' (ID: {supplier_id})")
    return

@app.post("/purchase_orders/", response_model=PurchaseOrder, status_code=201, tags=["Purchasing"])
def create_purchase_order(order: PurchaseOrderCreate, db: Session = Depends(get_db), current_user: User = Depends(inventory_access)):
    supplier = db.query(SupplierModel).filter(SupplierModel.id == order.supplier_id).first()
    if not supplier:
        raise HTTPException(status_code=404, detail="Supplier not found")
    try:
        with db.begin_nested():
            total_amount = 0
            order_items = []
            for item_data in order.items:
                total_amount += item_data.quantity * item_data.unit_price
                order_items.append(PurchaseOrderItemModel(**item_data.model_dump()))
            db_order = PurchaseOrderModel(supplier_id=order.supplier_id, order_date=order.order_date, status=order.status, expected_delivery_date=order.expected_delivery_date, total_amount=total_amount, items=order_items)
            db.add(db_order)
        db.commit()
        db.refresh(db_order)
        create_audit_log(db, current_user.email, f"Created purchase order ID {db_order.id} from {supplier.name}")
        return db_order
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/purchase_orders/", response_model=List[PurchaseOrder], tags=["Purchasing"])
def read_purchase_orders(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), current_user: User = Depends(inventory_access)):
    return db.query(PurchaseOrderModel).offset(skip).limit(limit).all()

@app.get("/purchase_orders/{order_id}", response_model=PurchaseOrder, tags=["Purchasing"])
def read_purchase_order(order_id: int, db: Session = Depends(get_db), current_user: User = Depends(inventory_access)):
    db_order = db.query(PurchaseOrderModel).filter(PurchaseOrderModel.id == order_id).first()
    if not db_order:
        raise HTTPException(status_code=404, detail="Purchase order not found")
    return db_order

@app.put("/purchase_orders/{order_id}", response_model=PurchaseOrder, tags=["Purchasing"])
def update_purchase_order(order_id: int, order_update: PurchaseOrderUpdate, db: Session = Depends(get_db), current_user: User = Depends(inventory_access)):
    db_order = db.query(PurchaseOrderModel).filter(PurchaseOrderModel.id == order_id).first()
    if not db_order:
        raise HTTPException(status_code=404, detail="Purchase order not found")
    old_status = db_order.status
    try:
        with db.begin_nested():
            update_data = order_update.model_dump(exclude_unset=True)
            for key, value in update_data.items():
                setattr(db_order, key, value)
            if old_status.lower() != "completed" and db_order.status and db_order.status.lower() == "completed":
                product_ids = [item.product_id for item in db_order.items]
                products = db.query(ProductModel).filter(ProductModel.id.in_(product_ids)).with_for_update().all()
                product_map = {p.id: p for p in products}
                for item in db_order.items:
                    product = product_map.get(item.product_id)
                    if product:
                        product.quantity_in_stock += item.quantity
        db.commit()
        db.refresh(db_order)
        create_audit_log(db, current_user.email, f"Updated purchase order ID {order_id} status to '{db_order.status}'")
        return db_order
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

def _get_invoice_with_balance(db: Session, invoice_id: int) -> Invoice:
    db_invoice = db.query(InvoiceModel).filter(InvoiceModel.id == invoice_id).first()
    if not db_invoice:
        return None
    total_paid = db.query(func.sum(PaymentModel.amount_paid)).filter(PaymentModel.invoice_id == invoice_id).scalar() or 0.0
    invoice_data = Invoice.model_validate(db_invoice)
    invoice_data.total_paid = total_paid
    invoice_data.balance_due = db_invoice.total_amount - total_paid
    return invoice_data

@app.post("/invoices/", response_model=Invoice, status_code=201, tags=["Finance"])
def create_invoice(invoice: InvoiceCreate, db: Session = Depends(get_db), current_user: User = Depends(finance_access)):
    db_invoice_data = {}
    if invoice.sales_order_id:
        db_order = db.query(SalesOrderModel).filter(SalesOrderModel.id == invoice.sales_order_id).first()
        if not db_order:
            raise HTTPException(status_code=404, detail="Sales order not found")
        if db_order.invoice:
            raise HTTPException(status_code=400, detail="Invoice already exists for this sales order")
        db_invoice_data = {"customer_id": db_order.customer_id, "sales_order_id": db_order.id, "total_amount": db_order.total_amount, "due_date": invoice.due_date}
    elif invoice.purchase_order_id:
        db_order = db.query(PurchaseOrderModel).filter(PurchaseOrderModel.id == invoice.purchase_order_id).first()
        if not db_order:
            raise HTTPException(status_code=404, detail="Purchase order not found")
        if db_order.invoice:
            raise HTTPException(status_code=400, detail="Invoice (bill) already exists for this purchase order")
        db_invoice_data = {"supplier_id": db_order.supplier_id, "purchase_order_id": db_order.id, "total_amount": db_order.total_amount, "due_date": invoice.due_date}
    else:
        raise HTTPException(status_code=400, detail="Must provide either sales_order_id or purchase_order_id")
    db_invoice = InvoiceModel(**db_invoice_data)
    db.add(db_invoice)
    db.commit()
    db.refresh(db_invoice)
    create_audit_log(db, current_user.email, f"Created invoice ID {db_invoice.id}")
    return _get_invoice_with_balance(db, db_invoice.id)

@app.get("/invoices/", response_model=List[Invoice], tags=["Finance"])
def read_invoices(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), current_user: User = Depends(finance_access)):
    db_invoices = db.query(InvoiceModel).offset(skip).limit(limit).all()
    return [_get_invoice_with_balance(db, db_invoice.id) for db_invoice in db_invoices]

@app.get("/invoices/{invoice_id}", response_model=Invoice, tags=["Finance"])
def read_invoice(invoice_id: int, db: Session = Depends(get_db), current_user: User = Depends(finance_access)):
    invoice_with_balance = _get_invoice_with_balance(db, invoice_id)
    if not invoice_with_balance:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return invoice_with_balance

@app.put("/invoices/{invoice_id}", response_model=Invoice, tags=["Finance"])
def update_invoice(invoice_id: int, invoice_update: InvoiceUpdate, db: Session = Depends(get_db), current_user: User = Depends(finance_access)):
    db_invoice = db.query(InvoiceModel).filter(InvoiceModel.id == invoice_id).first()
    if not db_invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    update_data = invoice_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_invoice, key, value)
    db.commit()
    db.refresh(db_invoice)
    create_audit_log(db, current_user.email, f"Updated invoice ID {invoice_id}")
    return _get_invoice_with_balance(db, invoice_id)

@app.post("/payments/", response_model=Payment, status_code=201, tags=["Finance"])
def create_payment(payment: PaymentCreate, db: Session = Depends(get_db), current_user: User = Depends(finance_access)):
    db_invoice = db.query(InvoiceModel).filter(InvoiceModel.id == payment.invoice_id).with_for_update().first()
    if not db_invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    if db_invoice.status == InvoiceStatus.paid:
        raise HTTPException(status_code=400, detail="Invoice is already fully paid")
    try:
        with db.begin_nested():
            db_payment = PaymentModel(**payment.model_dump())
            db.add(db_payment)
            db.flush()
            total_paid = db.query(func.sum(PaymentModel.amount_paid)).filter(PaymentModel.invoice_id == payment.invoice_id).scalar() or 0.0
            if total_paid >= db_invoice.total_amount:
                db_invoice.status = InvoiceStatus.paid
            elif total_paid > 0:
                db_invoice.status = InvoiceStatus.partially_paid
        db.commit()
        db.refresh(db_payment)
        create_audit_log(db, current_user.email, f"Recorded payment ID {db_payment.id} for invoice ID {payment.invoice_id}")
        return db_payment
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/payments/invoice/{invoice_id}", response_model=List[Payment], tags=["Finance"])
def read_payments_for_invoice(invoice_id: int, db: Session = Depends(get_db), current_user: User = Depends(finance_access)):
    if not db.query(InvoiceModel).filter(InvoiceModel.id == invoice_id).first():
        raise HTTPException(status_code=404, detail="Invoice not found")
    return db.query(PaymentModel).filter(PaymentModel.invoice_id == invoice_id).all()

@app.get("/dashboard/summary", tags=["Dashboard"])
def get_dashboard_summary(db: Session = Depends(get_db), current_user: User = Depends(viewer_access)):
    total_products = db.query(ProductModel).count()
    total_customers = db.query(CustomerModel).count()
    total_sales_orders = db.query(SalesOrderModel).count()
    total_revenue = db.query(func.sum(SalesOrderModel.total_amount)).scalar() or 0
    total_outstanding = db.query(func.sum(InvoiceModel.total_amount)).filter(InvoiceModel.status.in_([InvoiceStatus.sent, InvoiceStatus.overdue, InvoiceStatus.partially_paid])).scalar() or 0
    total_payments_received = db.query(func.sum(PaymentModel.amount_paid)).scalar() or 0
    return {"total_products": total_products, "total_customers": total_customers, "total_sales_orders": total_sales_orders, "total_revenue": f"{total_revenue:.2f}", "total_outstanding_ar": f"{total_outstanding:.2f}", "total_payments_received": f"{total_payments_received:.2f}"}

@app.get("/audit-logs/", response_model=List[AuditLog], tags=["Administration"])
def read_audit_logs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), current_user: User = Depends(admin_only)):
    return db.query(AuditLogModel).order_by(AuditLogModel.timestamp.desc()).offset(skip).limit(limit).all()

if __name__ == "__main__":
    if not os.path.exists(DATABASE_URL.split("///")[-1]):
        create_db_and_tables()
    if not SECRET_KEY or not REFRESH_SECRET_KEY:
        print("FATAL: SECRET_KEY and REFRESH_SECRET_KEY must be set in your .env file.")
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)
