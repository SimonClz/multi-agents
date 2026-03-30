import sqlite3
import json
import msgpack
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

conn = sqlite3.connect("conversations.db")
cursor = conn.cursor()
serde = JsonPlusSerializer()

def decode_value(value, value_type):
    if value is None:
        return None
    try:
        return serde.loads_typed((value_type, value))
    except:
        pass
    try:
        return msgpack.unpackb(value, raw=False)
    except:
        pass
    try:
        return json.loads(value)
    except:
        return repr(value)

def format_messages(content):
    if isinstance(content, list):
        result = []
        for item in content:
            if hasattr(item, 'type') and hasattr(item, 'content'):
                result.append(f"  [{item.type.upper()}]: {item.content}")
            elif isinstance(item, dict) and 'type' in item:
                result.append(f"  [{item['type'].upper()}]: {item.get('content', item)}")
        return "\n".join(result)
    return str(content)

# Récupère le dernier checkpoint par thread (état complet)
cursor.execute("""
    SELECT thread_id, type, checkpoint
    FROM checkpoints
    WHERE ROWID IN (
        SELECT MAX(ROWID) FROM checkpoints GROUP BY thread_id
    )
    ORDER BY thread_id
""")
rows = cursor.fetchall()

for thread_id, value_type, checkpoint_blob in rows:
    print(f"\n{'='*60}")
    print(f"🧵 CONVERSATION : {thread_id}")
    print(f"{'='*60}")

    checkpoint = decode_value(checkpoint_blob, value_type)

    if isinstance(checkpoint, dict):
        channel_values = checkpoint.get('channel_values', {})
        messages = channel_values.get('messages', [])
        if messages:
            print(format_messages(messages))
        else:
            print(json.dumps(channel_values, indent=2, ensure_ascii=False))
    else:
        print(str(checkpoint))

conn.close()