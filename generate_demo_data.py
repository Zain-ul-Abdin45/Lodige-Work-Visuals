"""
Generates illustrative (not real) daily KPI data for the talent-brand dashboard demo.

Output schema matches the tidy archive format used by the real ETL:
    date | platform | entity | metric | value

Run: python generate_demo_data.py
Writes: data/demo_kpis.csv
"""

import numpy as np
import pandas as pd

SEED = 42
DAYS = 120
END_DATE = pd.Timestamp.today().normalize()
START_DATE = END_DATE - pd.Timedelta(days=DAYS - 1)

rng = np.random.default_rng(SEED)
dates = pd.date_range(START_DATE, END_DATE, freq="D")
t = np.arange(len(dates))
weekday = dates.weekday.values  # 0=Mon ... 6=Sun

# Slow upward trend across the window (brand-building story), same shape reused per metric.
growth = 1 + 0.35 * (t / t.max())


def smooth(values, window=3):
    """Light centered rolling average so day-to-day noise doesn't dominate the trend."""
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy()


def daily(base, growth_curve, weekday_factor, noise_pct=0.12, floor=0):
    """base * growth * weekday seasonality * noise, lightly smoothed, rounded to whole units."""
    values = base * growth_curve * weekday_factor[weekday]
    noise = rng.normal(1.0, noise_pct, size=len(values))
    values = smooth(values * noise)
    return np.maximum(np.round(values), floor).astype(int)


# Weekday multipliers: B2B channels (site, LinkedIn) peak on weekdays; FB/IG a touch flatter.
B2B_WEEK = np.array([1.15, 1.2, 1.2, 1.15, 1.0, 0.6, 0.55])   # Mon..Sun
SOCIAL_WEEK = np.array([1.0, 1.05, 1.05, 1.0, 0.95, 0.9, 0.9])

rows = []


def add(platform, entity, metric, values):
    for d, v in zip(dates, values):
        rows.append((d.date().isoformat(), platform, entity, metric, int(v)))


# ---------------------------------------------------------------- WEBSITE (GA4)
sessions = daily(180, growth, B2B_WEEK, noise_pct=0.08)
engaged_sessions = np.minimum(sessions, daily(115, growth, B2B_WEEK, noise_pct=0.08))
total_users = np.minimum(sessions, daily(150, growth, B2B_WEEK, noise_pct=0.08))
new_users = np.minimum(total_users, daily(95, growth, B2B_WEEK, noise_pct=0.08))
screen_page_views = daily(260, growth, B2B_WEEK, noise_pct=0.08)
apply_page_views = daily(14, growth, B2B_WEEK, noise_pct=0.13)
avg_engagement_time_sec = daily(48, np.ones_like(growth), B2B_WEEK, noise_pct=0.08)

add("Website", "careers_site", "sessions", sessions)
add("Website", "careers_site", "engaged_sessions", engaged_sessions)
add("Website", "careers_site", "total_users", total_users)
add("Website", "careers_site", "new_users", new_users)
add("Website", "careers_site", "screen_page_views", screen_page_views)
add("Website", "careers_site", "apply_page_views", apply_page_views)
add("Website", "careers_site", "avg_engagement_time_sec", avg_engagement_time_sec)

# ---------------------------------------------------------------- LINKEDIN
impressions = daily(2200, growth, B2B_WEEK, noise_pct=0.08)
members_reached = np.minimum(impressions, daily(1500, growth, B2B_WEEK, noise_pct=0.1))
reactions = daily(55, growth, B2B_WEEK, noise_pct=0.13)
comments = daily(9, growth, B2B_WEEK, noise_pct=0.16)
reposts = daily(6, growth, B2B_WEEK, noise_pct=0.16)

add("LinkedIn", "company_page", "impressions", impressions)
add("LinkedIn", "company_page", "members_reached", members_reached)
add("LinkedIn", "company_page", "reactions", reactions)
add("LinkedIn", "company_page", "comments", comments)
add("LinkedIn", "company_page", "reposts", reposts)

# ---------------------------------------------------------------- FACEBOOK
fb_views = daily(1400, growth, SOCIAL_WEEK, noise_pct=0.2)
fb_reach = np.minimum(fb_views, daily(1100, growth, SOCIAL_WEEK, noise_pct=0.2))
fb_likes = daily(38, growth, SOCIAL_WEEK, noise_pct=0.13)
fb_comments = daily(5, growth, SOCIAL_WEEK, noise_pct=0.16)
fb_shares = daily(4, growth, SOCIAL_WEEK, noise_pct=0.16)
fb_link_clicks = daily(12, growth, SOCIAL_WEEK, noise_pct=0.16)
fb_leads = daily(2, growth, SOCIAL_WEEK, noise_pct=0.2)

add("Facebook", "page", "views", fb_views)
add("Facebook", "page", "reach", fb_reach)
add("Facebook", "page", "likes_reactions", fb_likes)
add("Facebook", "page", "comments", fb_comments)
add("Facebook", "page", "shares", fb_shares)
add("Facebook", "page", "link_clicks", fb_link_clicks)
add("Facebook", "page", "leads", fb_leads)

# ---------------------------------------------------------------- INSTAGRAM
ig_views = daily(1900, growth, SOCIAL_WEEK, noise_pct=0.2)
ig_reach = np.minimum(ig_views, daily(1450, growth, SOCIAL_WEEK, noise_pct=0.2))
ig_likes = daily(70, growth, SOCIAL_WEEK, noise_pct=0.13)
ig_comments = daily(7, growth, SOCIAL_WEEK, noise_pct=0.16)
ig_shares = daily(9, growth, SOCIAL_WEEK, noise_pct=0.16)
ig_link_clicks = daily(10, growth, SOCIAL_WEEK, noise_pct=0.16)
ig_leads = daily(3, growth, SOCIAL_WEEK, noise_pct=0.2)

add("Instagram", "account", "views", ig_views)
add("Instagram", "account", "reach", ig_reach)
add("Instagram", "account", "likes_reactions", ig_likes)
add("Instagram", "account", "comments", ig_comments)
add("Instagram", "account", "shares", ig_shares)
add("Instagram", "account", "link_clicks", ig_link_clicks)
add("Instagram", "account", "leads", ig_leads)

df = pd.DataFrame(rows, columns=["date", "platform", "entity", "metric", "value"])
df.to_csv("data/demo_kpis.csv", index=False)
print(f"Wrote {len(df)} rows for {df['platform'].nunique()} platforms, "
      f"{df['date'].min()} to {df['date'].max()} -> data/demo_kpis.csv")

# ---------------------------------------------------------------- AUDIENCE (period snapshot, not daily)
# LinkedIn reports reached-member seniority. Instagram (Business/Creator accounts)
# still reports reach by age and gender via instagram_manage_insights.
# Facebook Page Insights lost age/gender breakdowns in Meta's March 2024
# deprecation, so it isn't generated here - showing it would misrepresent
# what the live API can actually return.
# These are illustrative shares of the current audience, not a time series.

seniority = pd.DataFrame({
    "segment": ["Entry", "Senior", "Manager", "Director", "VP+"],
    "value": [18, 33, 26, 15, 8],
})
seniority.insert(0, "platform", "LinkedIn")
seniority.to_csv("data/demo_linkedin_seniority.csv", index=False)

age_gender_rows = []
# (age bracket, female %, male %)
instagram_age_gender = [("18-24", 14, 12), ("25-34", 21, 18), ("35-44", 11, 9), ("45-54", 5, 4), ("55+", 3, 3)]
for bracket, female, male in instagram_age_gender:
    age_gender_rows.append(("Instagram", bracket, "Female", female))
    age_gender_rows.append(("Instagram", bracket, "Male", male))

age_gender = pd.DataFrame(age_gender_rows, columns=["platform", "age_bracket", "gender", "value"])
age_gender.to_csv("data/demo_age_gender.csv", index=False)

print("Wrote data/demo_linkedin_seniority.csv and data/demo_age_gender.csv")
