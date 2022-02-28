import pymysql
import csv
import codecs


def get_conn():
    conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='ypq001207', db='insight', charset='utf8')
    return conn


def insert(cur, sql, args):
    cur.execute(sql, args)


def read_csv_to_mysql(filename, cnt_dimension):
    with codecs.open(filename=filename, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        conn = get_conn()
        cur = conn.cursor()
        table_name = filename.split('/')[1].split('.')[0]
        sql = 'insert into ' + table_name + ' values(%s'
        for i in range(cnt_dimension - 1):
            sql += ',%s'
        sql += ')'
        print(sql)
        for item in reader:
            if item[1] is None or item[1] == '':
                continue
            args = tuple(item)
            print(args)
            insert(cur, sql=sql, args=args)
        conn.commit()
        cur.close()
        conn.close()


if __name__ == '__main__':
    read_csv_to_mysql('../dataset/country.csv', 4)
