from app.models.tennis_predictor import get_default_tennis_predictor

def main():
    predictor = get_default_tennis_predictor()

    # Fake example: imagine p1 is stronger on most metrics
    features = {
        "d_last5_hold": 5.0,
        "d_last5_break": 3.0,
        "d_last10_hold": 4.0,
        "d_last10_break": 2.5,
        "d_surf_last10_hold": 6.0,
        "d_surf_last10_break": 3.0,
        "d_last10_aces_pg": 0.20,
        "d_surf_last10_aces_pg": 0.18,
        "d_last10_df_pg": -0.05,        # p1 has fewer DFs per game than p2
        "d_surf_last10_df_pg": -0.04,
        "d_last10_tb_match_rate": 5.0,
        "d_last10_tb_win_pct": 10.0,
        "d_surf_last10_tb_match_rate": 3.0,
        "d_surf_last10_tb_win_pct": 8.0,
    }

    prob = predictor.predict_proba(features)
    print("P(p1 wins) =", prob)

if __name__ == "__main__":
    main()
