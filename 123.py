
for i in range(10):
    for j in range(10):
        for m in range(10):
            # print(f'{i}, {j}, {m}')
            pass

rty = {}


def qwe(_rty, t=5, iter=range(10)):
    if t>0:
        for i in iter:
            _rty.update({str(i): {}})
            _rty[str(i)].update(qwe(_rty[str(i)], t-1))

    return _rty


rty = qwe(rty)
print(rty)
