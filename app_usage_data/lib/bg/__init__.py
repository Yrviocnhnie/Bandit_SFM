"""Task C — Background-App Suspension Prediction.

Given the set B(t) of apps currently in background at anchor time t, predict
for each a in B(t) the probability the user will foreground a in (t, t+H].
Suspension score = 1 - p_used; the OS kills the top-scored apps to free RAM.
"""
