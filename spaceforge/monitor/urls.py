from django.urls import path
from .views import *

urlpatterns = [
    path('p0', P0.as_view(), name = 'p0'),
    path('p1', P1.as_view(), name = 'p1'),
    path('p2', P2.as_view(), name = 'p2'),
    path('p3', P3.as_view(), name = 'p3'),
    path('p4', P4.as_view(), name = 'p4'),

]

