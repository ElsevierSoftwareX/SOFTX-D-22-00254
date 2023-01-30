#!/usr/bin/env python3
## ---------------------------------------------------------------------
## Copyright (C) 2021 - 2022 by the lifex authors.
##
## This file is part of lifex.
##
## lifex is free software; you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## lifex is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## Lesser General Public License for more details.

## You should have received a copy of the GNU Lesser General Public License
## along with lifex.  If not, see <http://www.gnu.org/licenses/>.
## ---------------------------------------------------------------------

# Author: Pasquale Claudio Africa <pasqualeclaudio.africa@polimi.it>.

# Retrieve the last available coverage information.

import gitlab
import os

gl = gitlab.Gitlab("https://gitlab.com", private_token = os.environ["LIFEX_READ_TOKEN"])
gl.auth()

project = gl.projects.get(os.environ["CI_PROJECT_ID"])

# Search for a real coverage computation through
# the last 5000 jobs run on the default branch.
for page in range(1, 50):
    jobs = project.jobs.list(all = False, page = page, per_page = 100)

    for job in jobs:
        if (job.status == "success"
            and job.ref == os.environ["CI_DEFAULT_BRANCH"]
            and job.name == "coverage"
            and job.coverage is not None
                and hasattr(job, "artifacts_file")):
            print("Job ID: {}".format(job.id))
            print("Last available coverage: {}%".format(job.coverage))
            f = open("artifacts.zip", "wb")
            job.artifacts(streamed = True, action = f.write)

            exit(0)

print("ERROR: no coverage information found.")
exit(1)
