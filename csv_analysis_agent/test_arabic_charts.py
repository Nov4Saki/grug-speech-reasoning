"""
اختبارات عربية لـ chart_tool — 5 سيناريوهات مختلفة
Arabic test suite for chart_tool using Egyptian Arabic queries.
"""
import sys
import io
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from graph import graph


def run(query: str, label: str, df: pd.DataFrame):
    print(f"\n{'='*65}")
    print(f"  🧪 {label}")
    print(f"  📝 {query}")
    print(f"{'='*65}")
    state = {
        "user_query": query,
        "dataframe": df.copy(),
        "plan": [],
        "current_step_index": 0,
        "result": None,
        "error": "",
    }
    final = graph.invoke(state)
    if final.get("error"):
        print(f"  ❌ Error: {final['error']}")
    else:
        print("  📋 الخطة:")
        for i, step in enumerate(final.get("plan", []), 1):
            print(f"     خطوة {i}: {step['tool_name']} ← {step['tool_args']}")
        print(f"  ✅ النتيجة: {final.get('result')}")


# ── تحميل البيانات ───────────────────────────────────────────────────────────
print("جاري تحميل البيانات …")
df = pd.read_excel("sales.xlsx")
print(f"تم التحميل: {len(df):,} صف × {len(df.columns)} عمود\n")

# ── اختبار 1: بار شارت — مجموع المبيعات لكل مدينة ───────────────────────────
run(
    query="ارسم بار شارت يوضح مجموع المبيعات لكل مدينة",
    label="Bar Chart — مجموع المبيعات لكل مدينة",
    df=df,
)

# ── اختبار 2: باي شارت — توزيع المبيعات على الفئات ──────────────────────────
run(
    query="عايز باي شارت يوضح نسبة مبيعات كل فئة من المنتجات",
    label="Pie Chart — نسبة المبيعات لكل فئة",
    df=df,
)

# ── اختبار 3: فلتر ثم رسم بياني — إلكترونيات فقط ────────────────────────────
run(
    query="فلتر منتجات الإلكترونيات بس، وبعدين ارسم بار شارت للمبيعات لكل مدينة",
    label="Filter → Bar Chart — إلكترونيات لكل مدينة",
    df=df,
)

# ── اختبار 4: بار شارت مجمع — المبيعات لكل فئة مقسمة على المدن ──────────────
run(
    query="ارسم بار شارت مجمع يوضح مبيعات كل فئة مقسمة حسب المدينة",
    label="Grouped Bar — الفئة × المدينة",
    df=df,
)

# ── اختبار 5: سكاتر بلوت — الكمية مقابل الإجمالي ────────────────────────────
run(
    query="ارسم سكاتر بلوت يوضح العلاقة بين الكمية والمبيعات الإجمالية ملون حسب الفئة",
    label="Scatter Plot — الكمية vs الإجمالي (ملون بالفئة)",
    df=df,
)

print("\n\nانتهت جميع الاختبارات ✅  — الرسوم البيانية محفوظة في مجلد ./charts/")
