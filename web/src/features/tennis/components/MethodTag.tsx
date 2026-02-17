import { CheckCircleOutlined, LineChartOutlined, ThunderboltOutlined } from '@ant-design/icons'
import { Tag } from 'antd'
import React from 'react'

interface MethodTagProps {
    numPredictors: number
}

const getMethodColor = (numPredictors: number): string => {
    if (numPredictors >= 3) return 'green'
    if (numPredictors === 2) return 'blue'
    return 'orange'
}

const getMethodIcon = (numPredictors: number): React.ReactNode => {
    if (numPredictors >= 3) return <CheckCircleOutlined />
    if (numPredictors === 2) return <ThunderboltOutlined />
    return <LineChartOutlined />
}

const MethodTag: React.FC<MethodTagProps> = ({ numPredictors }) => (
    <Tag color={getMethodColor(numPredictors)} icon={getMethodIcon(numPredictors)}>
        {numPredictors >= 3 ? 'ELO + Rolling + Market' : numPredictors === 2 ? 'Two predictors' : 'Single predictor'}
    </Tag>
)

export default MethodTag
