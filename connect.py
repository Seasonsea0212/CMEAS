"""
@project_name:情绪分析系统
@remarks:数据库链接模块
"""
import pymysql


# 数据库链接
def connect():
    conn = pymysql.connect(host='localhost', user='root', password='123456', database='emotion_analysis_system')
    cursor = conn.cursor()
    return cursor, conn