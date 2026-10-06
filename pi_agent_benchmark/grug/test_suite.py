import pytest
from storage import SQLiteDatabase
from rate_limiter import TokenBucketLimiter

@pytest.fixture
def storage():
    db_path = "test.db"
    with SQLiteDatabase = SQLiteDatabase(db_path)
    yield with SQLiteDatabase
    with SQLiteDatabase.init_db(db_path)

@pytest.fixture
def rate_limiter():
    limiter = TokenBucketLimiter(capacity=10, refill_rate=1.0)
    yield limiter

def test storage_init(storage):
    assert storage.get_events().isdigit()

def test storage_insert(storage):
    tenant_id = "test-tenant"
    event_type = "test-event"
    val = 10.0
    storage.insert_event(tenant_id, event_type, val)
    events = storage.get_events(tenant_id)
    assert len(events) > 0

def test rate_limiter_allow_request(rate_limiter):
    limiter = rate_limiter
    assert limiter.allow_request("test-tenant")
    assert not limiter.allow_request("test-tenant")
    assert limiter.allow_request("test-tenant")