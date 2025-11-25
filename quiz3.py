import threading
import random
import time
from datetime import datetime

try:
    # 외부 패키지 (허용될 경우에만 사용)
    import mysql.connector
except ImportError:
    mysql = None


class ParmSensor:
    """스마트 팜 센서 클래스"""

    def __init__(self, name):
        self.name = name
        self.temperature = 0
        self.light = 0
        self.humidity = 0
        self.lock = threading.Lock()

    def set_data(self):
        """센서 값을 랜덤 범위 안에서 설정한다."""
        with self.lock:
            self.temperature = random.randint(20, 30)
            self.light = random.randint(5000, 10000)
            self.humidity = random.randint(40, 70)

    def get_data(self):
        """현재 센서 값을 반환한다."""
        with self.lock:
            return self.temperature, self.light, self.humidity


class SensorQueue:
    """FIFO 큐 구현 (sensorQ)"""

    def __init__(self, max_size=0):
        self._items = []
        self._lock = threading.Lock()
        self._max_size = max_size

    def enqueue(self, item):
        """큐의 뒤에 항목을 추가한다."""
        with self._lock:
            if self._max_size and len(self._items) >= self._max_size:
                # 가장 오래된 데이터 제거 (선택 사항)
                self._items.pop(0)
            self._items.append(item)

    def dequeue(self):
        """큐의 앞에서 항목을 하나 꺼낸다. 비어 있으면 None."""
        with self._lock:
            if self._items:
                return self._items.pop(0)
            return None

    def is_empty(self):
        """큐가 비어 있는지 확인한다."""
        with self._lock:
            return len(self._items) == 0

    def size(self):
        """큐의 크기를 반환한다."""
        with self._lock:
            return len(self._items)


class SimpleDataFrame:
    """
    외부 라이브러리 없이 사용하는 매우 단순한 DataFrame 비슷한 객체.
    rows는 dict의 리스트로 구성한다.
    """

    def __init__(self):
        self.rows = []
        self.lock = threading.Lock()

    def add_row(self, row):
        with self.lock:
            self.rows.append(row)

    def get_rows_copy(self):
        """현재까지의 데이터를 복사해서 반환한다."""
        with self.lock:
            return list(self.rows)


# 전역 객체
sensor_q = SensorQueue()
data_frame = SimpleDataFrame()
stop_event = threading.Event()


# ------------------------ MySQL 관련 함수들 ------------------------ #

def get_db_connection():
    """
    MySQL 연결을 반환한다.
    실제 사용 시 자신의 DB 정보에 맞게 수정해야 한다.
    외부 패키지(mysql-connector-python)가 필요하다.
    """
    if mysql is None:
        # 과제 조건상 외부 패키지를 못 쓰는 경우 이 부분은 의사 코드로만 사용
        raise RuntimeError('mysql-connector-python 패키지가 필요합니다.')

    conn = mysql.connector.connect(
        host='localhost',
        user='your_user',
        password='your_password',
        database='your_database',
    )
    return conn


def create_table_if_not_exists():
    """
    parm_data 테이블을 생성한다.
    필드:
        id (int, primary key, auto increment)
        input_time (datetime)
        temperature (int)
        light (int)
        humidity (int)
        sensor_name (varchar)  # 보너스 과제용으로 추가
    """
    if mysql is None:
        print('MySQL 드라이버가 없어 테이블 생성은 건너뜀.')
        return

    conn = get_db_connection()
    cursor = conn.cursor()
    sql = (
        'CREATE TABLE IF NOT EXISTS parm_data ('
        'id INT AUTO_INCREMENT PRIMARY KEY,'
        'input_time DATETIME NOT NULL,'
        'temperature INT NOT NULL,'
        'light INT NOT NULL,'
        'humidity INT NOT NULL,'
        'sensor_name VARCHAR(20) NOT NULL'
        ')'
    )
    cursor.execute(sql)
    conn.commit()
    cursor.close()
    conn.close()


def insert_sensor_data(input_time, temperature, light, humidity, sensor_name):
    """
    센서 데이터를 parm_data 테이블에 입력한다.
    """
    if mysql is None:
        # 실제 과제 환경에서는 여기에서 MySQL에 insert 하도록 구현하면 된다.
        # 지금은 외부 패키지 제한 때문에 단순 출력으로 대체 가능.
        print(
            '[DEBUG] insert_sensor_data 호출 (DB 드라이버 없음): '
            f'{input_time}, {sensor_name}, {temperature}, {light}, {humidity}'
        )
        return

    conn = get_db_connection()
    cursor = conn.cursor()
    sql = (
        'INSERT INTO parm_data (input_time, temperature, light, humidity, sensor_name) '
        'VALUES (%s, %s, %s, %s, %s)'
    )
    cursor.execute(sql, (input_time, temperature, light, humidity, sensor_name))
    conn.commit()
    cursor.close()
    conn.close()


def get_sensor_data():
    """
    parm_data 테이블에서 모든 데이터를 가져온다.
    반환 형식: 리스트[dict]
    """
    if mysql is None:
        print('MySQL 드라이버가 없어 get_sensor_data는 샘플 데이터로 대체합니다.')
        # 샘플 데이터 (그래프/통계 테스트용)
        now = datetime.now()
        return [
            {
                'input_time': now,
                'temperature': 25,
                'light': 8000,
                'humidity': 60,
                'sensor_name': 'Parm-1',
            },
            {
                'input_time': now,
                'temperature': 27,
                'light': 9000,
                'humidity': 65,
                'sensor_name': 'Parm-2',
            },
        ]

    conn = get_db_connection()
    cursor = conn.cursor()
    sql = (
        'SELECT input_time, temperature, light, humidity, sensor_name '
        'FROM parm_data ORDER BY input_time'
    )
    cursor.execute(sql)

    rows = []
    for row in cursor:
        rows.append(
            {
                'input_time': row[0],
                'temperature': row[1],
                'light': row[2],
                'humidity': row[3],
                'sensor_name': row[4],
            }
        )

    cursor.close()
    conn.close()
    return rows


# ------------------------ 쓰레드 함수들 ------------------------ #

def sensor_worker(sensor):
    """
    각 센서별로 10초마다 데이터를 생성하고 출력하며,
    sensorQ와 DataFrame에 저장한다.
    """
    while not stop_event.is_set():
        sensor.set_data()
        temperature, light, humidity = sensor.get_data()
        now = datetime.now()
        ts_str = now.strftime('%Y-%m-%d %H:%M:%S')

        # 출력 형식: 2000-08-01 14:30:30 Parm1 — temp 00, light 000, humi 00
        # 여기서는 실제 현재 시간/값 사용
        print(
            f'{ts_str} {sensor.name} — temp {temperature}, '
            f'light {light}, humi {humidity}'
        )

        # DataFrame(유사) 에도 저장
        row = {
            'input_time': now,
            'sensor_name': sensor.name,
            'temperature': temperature,
            'light': light,
            'humidity': humidity,
        }
        data_frame.add_row(row)

        # 큐에 저장 (DB 쓰레드가 소비)
        sensor_q.enqueue(row)

        # 10초 대기
        time.sleep(10)


def queue_to_db_worker():
    """
    1초마다 sensorQ를 확인해서,
    데이터가 있으면 FIFO 순서대로 꺼내서 DB에 insert 한다.
    """
    while not stop_event.is_set():
        while not sensor_q.is_empty():
            row = sensor_q.dequeue()
            if row is None:
                break
            insert_sensor_data(
                row['input_time'],
                row['temperature'],
                row['light'],
                row['humidity'],
                row['sensor_name'],
            )
        time.sleep(1)


def five_min_average_worker():
    """
    보너스 과제:
    5분 단위로 DataFrame에 담긴 데이터의 평균을 계산해서 출력.
    (외부 라이브러리 없이 간단 구현)
    """
    last_index = 0
    while not stop_event.is_set():
        time.sleep(300)  # 5분

        rows = data_frame.get_rows_copy()
        if last_index >= len(rows):
            continue

        new_rows = rows[last_index:]
        last_index = len(rows)

        if not new_rows:
            continue

        # 센서별 평균 온도/조도/습도 계산
        sums = {}
        counts = {}
        for row in new_rows:
            name = row['sensor_name']
            if name not in sums:
                sums[name] = {'temperature': 0, 'light': 0, 'humidity': 0}
                counts[name] = 0
            sums[name]['temperature'] += row['temperature']
            sums[name]['light'] += row['light']
            sums[name]['humidity'] += row['humidity']
            counts[name] += 1

        print('===== 5분 평균 데이터 =====')
        for name in sums:
            c = counts[name]
            avg_temp = sums[name]['temperature'] / c
            avg_light = sums[name]['light'] / c
            avg_humi = sums[name]['humidity'] / c
            print(
                f'{name} -> avg_temp: {avg_temp:.1f}, '
                f'avg_light: {avg_light:.1f}, avg_humi: {avg_humi:.1f}'
            )
        print('==========================')


# ------------------------ 분석/그래프(텍스트) 관련 ------------------------ #

def print_temperature_graph():
    """
    get_sensor_data()로 가져온 값을 바탕으로
    센서별 시간별 온도 평균을 텍스트 그래프로 출력한다.
    보너스: 습도가 90%가 넘는 포인트마다 별(*)로 표시.
    (실제 데이터에서는 40~70 범위라 안 나올 수도 있지만, 로직만 구현)
    """
    rows = get_sensor_data()
    if not rows:
        print('그래프를 그릴 데이터가 없습니다.')
        return

    # 센서별, 시간(시 단위)별 평균 온도 계산
    stats = {}
    humi_over_90_points = []

    for row in rows:
        sensor_name = row['sensor_name']
        hour = row['input_time'].strftime('%Y-%m-%d %H:00')
        key = (sensor_name, hour)

        if key not in stats:
            stats[key] = {'temp_sum': 0, 'count': 0}
        stats[key]['temp_sum'] += row['temperature']
        stats[key]['count'] += 1

        if row['humidity'] >= 90:
            humi_over_90_points.append((sensor_name, row['input_time']))

    print('===== 센서별 시간별 평균 온도 (텍스트 그래프) =====')
    for key in sorted(stats):
        sensor_name, hour = key
        temp_avg = stats[key]['temp_sum'] / stats[key]['count']
        # 단순 텍스트 그래프 (온도만큼 # 출력)
        bar = '#' * int(temp_avg)
        print(f'{sensor_name} {hour}  {temp_avg:.1f}°C {bar}')
    print('===============================================')

    if humi_over_90_points:
        print('습도 90% 이상 지점(보너스 표시):')
        for name, t in humi_over_90_points:
            print(f'* {t} {name} (humidity >= 90%)')
    else:
        print('습도 90% 이상 포인트는 없습니다(또는 데이터상 없음).')


# ------------------------ main ------------------------ #

def main():
    # (선택) MySQL 테이블 생성
    create_table_if_not_exists()

    # 센서 5개 생성
    sensors = []
    for i in range(1, 6):
        name = f'Parm-{i}'
        sensors.append(ParmSensor(name))

    # 센서 쓰레드 시작
    sensor_threads = []
    for sensor in sensors:
        t = threading.Thread(target=sensor_worker, args=(sensor,))
        t.daemon = True
        t.start()
        sensor_threads.append(t)

    # 큐 -> DB 쓰레드 시작
    db_thread = threading.Thread(target=queue_to_db_worker)
    db_thread.daemon = True
    db_thread.start()

    # 5분 평균 쓰레드 (보너스) 시작
    avg_thread = threading.Thread(target=five_min_average_worker)
    avg_thread.daemon = True
    avg_thread.start()

    try:
        # 예시로 1분 동안만 돌린 뒤 종료
        # 실제 과제에서는 무한 루프 형태로 두고 Ctrl+C로 종료해도 된다.
        time.sleep(60)
    except KeyboardInterrupt:
        print('사용자에 의해 종료 요청')
    finally:
        stop_event.set()
        # 잠시 대기하여 쓰레드들이 마무리하도록 함
        time.sleep(2)

        # 종료 시 DB 데이터 기반 그래프 출력 (텍스트)
        print_temperature_graph()


if __name__ == '__main__':
    main()
