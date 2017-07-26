
# Создадим пустой словать Capitals
def anketa(gender, age, smoking):
    q, m, r, d = 0, 0, 0, 0
    Capitals = dict()
    if gender == 'M':
        d = d + 0.7
        round(m, 2)
        if 0 <= int(age) <= 49:
            m = d - 0.5
            round(m, 2)
        elif 50 <= int(age) <= 59:
            m = d
            round(m, 2)
        elif 60 <= int(age) <= 69:
            m = d + 1.2
            round(m, 2)
        elif int(age) >= 70:
            m = d + 5.5
            round (m, 2)
    elif gender == 'F':
        d = d + 0.6
        if 0 <= int(age) <= 49:
            m = d - 0.4
            round(m, 2)
        elif 50 <= int(age) <= 59:
            m = d - 0.1
            round(m, 2)
        elif 60 <= int(age) <= 69:
            m = d + 0.9
            round(m, 2)
        elif int(age) >= 70:
            m = d + 4.2
            round(m, 2)

    if smoking == 'yes':
        r = 0.8
    elif smoking == 'no':
        r = 0,2

    Capitals['gender'] = d
    Capitals['age'] = round(m, 2)
    Capitals['smoking'] = r
    return (Capitals)
