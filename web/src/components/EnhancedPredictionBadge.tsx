// FILE: web/src/components/tennis/EnhancedPredictionBadge.tsx
import React from 'react'
import { Tag, Tooltip } from 'antd'
import { EnhancedPredictionInputs } from '~/types/tennis'

interface Props {
    inputs: EnhancedPredictionInputs
}

export const EnhancedPredictionBadge: React.FC<Props> = ({ inputs }) => {
    const { method, num_predictors } = inputs

    const methods = method?.split('+') || []
    const predictors = num_predictors ?? 0

    const individual = inputs.individual_predictions
    const predictorObj = !Array.isArray(individual) && individual ? individual : {}
    const elo = predictorObj.elo
    const xgbSimple = predictorObj.xgb_simple
    const market = predictorObj.market

    // Color based on number of predictors
    const badgeColor = predictors === 3 ? 'green' : predictors === 2 ? 'blue' : 'orange'

    const tooltipContent = (
            <div className="text-xs">
            <div className="font-semibold mb-2">Ensemble Components:</div>
            {elo != null && <div>ELO: {(elo * 100).toFixed(1)}%</div>}
            {xgbSimple != null && <div>XGBoost: {(xgbSimple * 100).toFixed(1)}%</div>}
            {market != null && <div>Market: {(market * 100).toFixed(1)}%</div>}
            <div className="mt-2 text-gray-400">
                {predictors} predictor{predictors > 1 ? 's' : ''} active
            </div>
        </div>
    )

    return (
        <Tooltip title={tooltipContent}>
            <Tag color={badgeColor} className="text-xs">
                {methods.map((m) => m.toUpperCase()).join(' + ')}
            </Tag>
        </Tooltip>
    )
}
