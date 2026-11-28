# -*- coding: utf-8 -*-
"""Project-wide pytest setup.

Runs before any test module imports the app. The rate limiter is turned off
for the suite because hundreds of requests come from one test client; the
limiter has its own tests that switch it on explicitly.
"""

import os

os.environ["RATE_LIMIT_ENABLED"] = "false"
