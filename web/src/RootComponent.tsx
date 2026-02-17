import 'antd/dist/reset.css'
import React from 'react'
import { Route, BrowserRouter as Router, Routes } from 'react-router-dom'
import Navbar from './components/Navbar'
import './index.css'
import HomePage from './pages/HomePage'
import NotFoundPage from './pages/NotFoundPage'
import TennisEnhancedPredictionsPage from './pages/TennisEnhancedPredictionsPage'
import TennisPlayerStatsPage from './pages/TennisPlayerStatsPage'
import TennisPlayerComparePage from './pages/TennisPlayerComparePage'
import { ROUTES } from './resources/routes-constants'
import TennisParlaysPage from './pages/TennisParlaysPage'

const RootComponent: React.FC = () => {
    return (
        <Router>
            <Navbar />
            <Routes>
                <Route path="*" element={<NotFoundPage />} />
                <Route path={ROUTES.HOMEPAGE_ROUTE} element={<HomePage />} />
                <Route path={ROUTES.TENNIS_PARLAYS_ROUTE} element={<TennisParlaysPage />} />

                <Route path={ROUTES.TENNIS_ENHANCED_PREDICTIONS_ROUTE} element={<TennisEnhancedPredictionsPage />} />
                <Route path={ROUTES.TENNIS_PLAYER_STATS_ROUTE} element={<TennisPlayerStatsPage />} />
                <Route path={ROUTES.TENNIS_PLAYER_COMPARE_ROUTE} element={<TennisPlayerComparePage />} />
            </Routes>
        </Router>
    )
}

export default RootComponent
