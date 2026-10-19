-- データベースの基礎（2026年度） サンプルデータ sample.sql
-- PostgreSQL（Docker の pg18）と SQLite（Colab）の両方でそのまま動く
-- 使い方：pgAdmin のクエリツールでこのファイルを開いて実行（F5）する．
--        何度実行してもよい（最初に表を作り直すので，データは初期状態に戻る）

DROP TABLE IF EXISTS 供給;
DROP TABLE IF EXISTS 従業員;
DROP TABLE IF EXISTS 業者;
DROP TABLE IF EXISTS 部品;
DROP TABLE IF EXISTS 部門;

CREATE TABLE 部門 (
    部門番号  INTEGER PRIMARY KEY,
    部門名    TEXT NOT NULL);
CREATE TABLE 従業員 (
    従業員番号  INTEGER PRIMARY KEY,
    部門番号  INTEGER NOT NULL,
    氏名  TEXT NOT NULL,
    住所  TEXT,
    職級  INTEGER);
CREATE TABLE 部品 (
    部品番号  INTEGER PRIMARY KEY,
    部品名    TEXT NOT NULL);
CREATE TABLE 業者 (
    業者番号  INTEGER PRIMARY KEY,
    業者名  TEXT NOT NULL,
    住所  TEXT,
    電話番号  TEXT);
CREATE TABLE 供給 (
    部門番号  INTEGER NOT NULL,
    部品番号  INTEGER NOT NULL,
    業者番号  INTEGER NOT NULL,
    単価  INTEGER NOT NULL,
    数量    INTEGER NOT NULL);

INSERT INTO 部門 VALUES (1, '営業部');
INSERT INTO 部門 VALUES (2, '開発部');

INSERT INTO 従業員 VALUES (1, 1, '佐藤一郎', '横浜市', 2);
INSERT INTO 従業員 VALUES (2, 1, '田中次郎', '静岡市', 1);
INSERT INTO 従業員 VALUES (3, 2, '鈴木三郎', '横浜市', 2);
INSERT INTO 従業員 VALUES (4, 2, '山田四郎', '東京都', 1);

INSERT INTO 部品 VALUES (1, 'CPU');
INSERT INTO 部品 VALUES (2, 'HDD');
INSERT INTO 部品 VALUES (3, 'SSD');
INSERT INTO 部品 VALUES (4, 'ビデオカード');

INSERT INTO 業者 VALUES (1, 'インテル', 'アメリカ合衆国', '000-000-0001');
INSERT INTO 業者 VALUES (2, 'AMD', 'アメリカ合衆国', '000-000-0002');
INSERT INTO 業者 VALUES (3, 'Western Digital', 'アメリカ合衆国', '000-000-0003');

INSERT INTO 供給 VALUES (2, 1, 1, 10000, 100);
INSERT INTO 供給 VALUES (2, 1, 2,  9000,  50);
INSERT INTO 供給 VALUES (2, 2, 3,  5000, 150);
INSERT INTO 供給 VALUES (2, 3, 1,  6000,  60);
INSERT INTO 供給 VALUES (2, 3, 3,  5000,  90);
INSERT INTO 供給 VALUES (2, 4, 2, 15000, 150);
