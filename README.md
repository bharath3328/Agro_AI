# AgroAI - Plant Disease Detection System

A modern web application for detecting plant diseases using Artificial Intelligence.

## Features
- **AI-Powered Analysis**: Upload plant leaf images to detect diseases instantly (Few-Shot Learning).
- **Admin Dashboard**: Train new diseases dynamically and manage users.
- **My History**: Track past predictions and disease trends.
- **Secure Auth**: JWT-based authentication with email verification.

## Tech Stack
- **Frontend**: React, Vite, TailwindCSS
- **Backend**: Python, FastAPI, SQLAlchemy
- **Database**: MySQL (Production) or SQLite (Dev)
- **AI/ML**: PyTorch, ResNet/Encoder (Prototypical Networks)

## Setup Instructions

### 1. Backend
```bash
cd backend
# Setup .env file with DB & Email credentials
python app.py
```
Server runs at: `http://localhost:8000`

### 2. Frontend
```bash
cd frontend
npm install
npm run dev
```
App runs at: `http://localhost:3000`

## Author
AgroAI Team - Final Year Project
