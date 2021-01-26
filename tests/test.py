#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:35:25 2021

@author: ghiggi
"""
import unittest


class TestFoo(unittest.TestCase):
    """Fake test class in order to setup the tests module
    """

    def test_foo(self):
        """Fake test method in order to setup the test module
        """
        self.assertTrue(True)