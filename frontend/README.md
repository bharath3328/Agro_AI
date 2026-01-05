# AgroAI Frontend

Modern React frontend for the AgroAI Plant Disease Detection System.

## Features

- ✅ **Authentication** - Login and Registration
- ✅ **Image Upload** - Drag & drop interface for plant images
- ✅ **Disease Detection** - Real-time disease prediction
- ✅ **Grad-CAM Visualization** - Visual explanation of AI predictions
- ✅ **AI Advisory** - Intelligent disease recommendations
- ✅ **Prediction History** - View all past predictions
- ✅ **Responsive Design** - Works on all devices

## Tech Stack

- **React 18** - UI framework
- **Vite** - Build tool
- **React Router** - Navigation
- **Tailwind CSS** - Styling
- **Axios** - API client
- **React Dropzone** - File upload
- **Lucide React** - Icons
- **React Hot Toast** - Notifications

## Setup

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Start Development Server

```bash
npm run dev
```

The frontend will be available at: http://localhost:3000

### 3. Build for Production

```bash
npm run build
```

## Configuration

Make sure your backend is running on `http://localhost:8000` (default).

The frontend is configured to proxy API requests to the backend via Vite's proxy (see `vite.config.js`).

## Project Structure

```
frontend/
├── src/
│   ├── components/     # Reusable components
│   ├── contexts/       # React contexts (Auth)
│   ├── pages/          # Page components
│   ├── services/       # API services
│   ├── App.jsx         # Main app component
│   ├── main.jsx        # Entry point
│   └── index.css       # Global styles
├── index.html
├── package.json
├── vite.config.js
└── tailwind.config.js
```

## Pages

- **Login** (`/login`) - User authentication
- **Register** (`/register`) - New user registration
- **Dashboard** (`/`) - Main page with image upload
- **History** (`/history`) - Prediction history
- **Prediction Detail** (`/prediction/:id`) - Detailed prediction results with Grad-CAM

## API Integration

The frontend communicates with the backend API at `/api/v1`:

- `POST /api/v1/auth/login` - Login
- `POST /api/v1/auth/register` - Register
- `GET /api/v1/auth/me` - Get current user
- `POST /api/v1/diagnosis/predict` - Predict disease
- `GET /api/v1/diagnosis/history` - Get prediction history
- `GET /api/v1/diagnosis/history/:id` - Get prediction details
- `GET /api/v1/diagnosis/gradcam/:id` - Get Grad-CAM image

## Environment Variables

No environment variables needed for development. The frontend uses Vite's proxy to connect to the backend.

For production, you may want to set:
- `VITE_API_URL` - Backend API URL (defaults to `/api/v1`)

## Development

The app uses:
- **Hot Module Replacement (HMR)** - Instant updates during development
- **React Fast Refresh** - Preserves component state on updates
- **Tailwind JIT** - Just-in-time CSS compilation

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)
