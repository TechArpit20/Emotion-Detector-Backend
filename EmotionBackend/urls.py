
from django.contrib import admin
from .views import *
from django.urls import path
from rest_framework.routers import DefaultRouter
from django.conf import settings
from django.conf.urls.static import static

# router= DefaultRouter()
# router.register('image', DetectView, basename='image')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('login',LoginView.as_view()),
    path('signup',SignUpView.as_view()),
    path('image',DetectView.as_view()),
    path('text',TextView.as_view()),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root= settings.MEDIA_ROOT)