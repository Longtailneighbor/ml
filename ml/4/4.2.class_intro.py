#!/usr/bin/python
# -*- coding:utf-8 -*-

class People:
    def __init__(self, name, age, score):
        self.name = name
        self.age = age
        self.__score = score
        self.print_people()

    def print_people(self):
        str = u'%s的年龄:%d，成绩为:%.2f' % (self.name, self.age, self.__score)
        print str

    __print_people = print_people


class Student(People):
    def __init__(self, name, age, score):
        People.__init__(self, name, age, score)
        self.name = 'Student ' + self.name

    def print_people(self):
        str = u'%s的年龄：%d' % (self.name, self.age)
        print str


def func(p):
    p.age = 11


if __name__ == '__main__':
    people = People('Tom', 10, 3.14159)
    func(people)
    people.print_people()
    print

    student = Student('Jerry', 12, 2.71825)
    print

    people.print_people()
    student.print_people()

    People.print_people(people)
    People.print_people(student)
