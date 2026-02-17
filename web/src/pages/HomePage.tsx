import React from 'react'
import { useNavigate } from 'react-router-dom'
import { Button } from 'antd'
import DateDisplay from '../components/DateDisplay'
import { ROUTES } from '../resources/routes-constants'

const HomePage: React.FC = () => {
    const navigate = useNavigate()

    return (
        <div className="relative w-full min-h-screen flex flex-col items-center justify-center gap-6 bg-white">
            <h1 className="text-7xl text-gray-900">Hello world!</h1>
            <DateDisplay />

            <Button type="primary" onClick={() => navigate(ROUTES.TENNIS_ENHANCED_PREDICTIONS_ROUTE)}>
                Tennis predictions today
            </Button>
        </div>
    )
}

export default HomePage
