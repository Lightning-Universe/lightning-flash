# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import wraps


def retry(max_retries: int):
    """Decorator to retry a flaky test a fix number of times before failing.

    Note: This does not implement any form of backoff so should not be use for tests which are flaky due to network
    issues.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    func(*args, **kwargs)
                except AssertionError as e:
                    if i == max_retries - 1:
                        raise e
                    continue
                break

        return wrapper

    return decorator
