// web/src/components/Navbar.tsx
import React from 'react'
import { Menu } from 'antd'
import { useNavigate, useLocation } from 'react-router-dom'
import { DollarOutlined, ThunderboltOutlined } from '@ant-design/icons'
import { ROUTES } from '../resources/routes-constants'

const Navbar: React.FC = () => {
    const navigate = useNavigate()
    const location = useLocation()

    const menuItems = [
        {
            key: ROUTES.TENNIS_ENHANCED_PREDICTIONS_ROUTE,
            icon: <ThunderboltOutlined />,
            label: 'Enhanced Predictions'
        },
        {
            key: ROUTES.TENNIS_PARLAYS_ROUTE,
            icon: <DollarOutlined />,
            label: 'Parlays'
        }
    ]

    return (
        <Menu
            mode="horizontal"
            selectedKeys={[location.pathname]}
            items={menuItems}
            onClick={({ key }) => navigate(key)}
            theme="dark"
            style={{
                position: 'sticky',
                top: 0,
                zIndex: 1000,
                width: '100%',
                borderBottom: '1px solid #374151'
            }}
        />
    )
}

export default Navbar
