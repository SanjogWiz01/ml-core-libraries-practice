from collections import deque
q = deque(["job1", "job2"])
q.append("job3")
print(q.popleft())
q.appendleft("priority_job")
print(q)
