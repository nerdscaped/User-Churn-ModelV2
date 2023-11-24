WITH beacons AS (
-- Download beacons w/metadata for later steps
SELECT *,  
       CASE WHEN DENSE_RANK() OVER (PARTITION BY datepartition ORDER BY daily_brand_plays DESC) <= 20 THEN 1 END AS top20_brands_indicator -- an indicator to identify populae plays (those of top 20 brands)
FROM 
(SELECT b.*,
       thing_custom_brand_id[1] brand,
       thing_custom_subcategory_id[1] subcat,
       COUNT(CASE WHEN action = 'play' THEN 1 END) OVER (PARTITION BY datepartition, thing_custom_brand_id[1]) AS daily_brand_plays,
       HOUR(CAST(SUBSTRING(REPLACE(b.timestamp_initiated,'T',' '),1,19) AS timestamp)) hour_timestamp
FROM awsdatacatalog.productionuktvdatabase.beacon b
LEFT JOIN awsdatacatalog.productionuktvdatabase.metadata m ON m.thing_id = b.thing_id
WHERE CAST(datepartition AS date) < current_date AND 
      b.user_primaryid IS NOT NULL AND 
      TRIM(b.user_primaryid) <> '' AND
      b.user_primaryid != 'null')),
      
days AS (
-- This step takes a random sample of days for the filter in the last step - this is to reduce the size of the dataset
SELECT datepartition 
FROM
(SELECT CAST(datepartition AS date) datepartition, ROW_NUMBER() OVER (ORDER BY CRC32(CAST(datepartition AS varbinary))) crc_row
FROM
(SELECT DISTINCT datepartition
FROM beacons
WHERE CAST(datepartition AS date) >= DATE_ADD('month', -16, current_date) AND 
      CAST(datepartition AS date) < DATE_ADD('day', -60, current_date)))
WHERE crc_row <= 200),

users_with_stats AS (
-- This gets users from user_table_firsts and adds some metadata e.g. if they came for a popular brand and their first login date
-- Train users - these are users used to train the model (i.e. have had the opportunity to churn) - 200,000 users are included in this by default
-- Predict users - these are the users we are trying to predict whether they will churn. These are users who have accessed within the last 60 days. Users who have accessed within a week are removed 
SELECT user_primaryid,
       date_first_login,
       first_brand_ranking_indicator
FROM
(SELECT user_primaryid,
       date_first_login,
       user_type,
       ROW_NUMBER() OVER (PARTITION BY user_type) AS row,
       CASE WHEN DENSE_RANK() OVER (PARTITION BY date_first_login ORDER BY first_brand_plays DESC) <= 10 THEN 1 ELSE 0 END first_brand_ranking_indicator -- This identifies if they came from a top 10 popular brand
       FROM
        (SELECT f.user_primaryid,
                CASE WHEN CAST(SUBSTRING(timestamp_last_login, 1, 10) AS date) >= DATE_ADD('day', -60, current_date) THEN 'predict' ELSE 'train' END user_type,
                first_brand_played,        
                CAST(SUBSTRING(timestamp_first_login, 1, 10) AS date) date_first_login,
                COUNT(*) OVER (PARTITION BY first_brand_played, SUBSTRING(timestamp_first_login, 1, 10)) first_brand_plays
         FROM awsdatacatalog.productionuktvdatabase.user_table_firsts f
         LEFT JOIN awsdatacatalog.productionuktvdatabase.user_table_sub_stats s ON f.user_primaryid = s.user_primaryid
         WHERE CAST(SUBSTRING(timestamp_first_login, 1, 10) AS date) >= DATE_ADD('month', -16, current_date) AND 
               CAST(SUBSTRING(timestamp_last_login, 1, 10) AS date) <= DATE_ADD('day', -7, current_date) AND 
               f.user_primaryid IS NOT NULL AND
               f.user_primaryid <> 'null' AND
               TRIM(f.user_primaryid) <> ''))
-- WHERE row <= 200000 OR user_type = 'predict'
),

user_access AS (
-- This gets all access points for each user in the step above
SELECT a.user_primaryid, 
       first_brand_ranking_indicator,
       date_first_login,
       CAST(datepartition AS date) access_date
FROM users_with_stats a
LEFT JOIN beacons b ON b.user_primaryid = a.user_primaryid
GROUP BY 1, 2, 3, 4),

users_last_actions AS ( 
-- This gets a detailed activity account for each access point
SELECT user_primaryid,  
       user_type,
       MAX(access_date) OVER (PARTITION BY user_primaryid) max_access,
       DATE_DIFF('day', LAG(access_date) OVER (PARTITION BY user_primaryid ORDER BY access_date ASC), access_date) days_last_access,
       access_date,
       DATE_DIFF('day', date_first_login, access_date) days_first_access,
       month_access_date,
       first_brand_ranking_indicator,
       plays, 
       t20_plays,
       actions,
       plays_L60D, 
       t20_plays_L60D,
       recs_L60D,
       actions_L60D,
       brands_played_L60D, 
       subcats_played_L60D, 
       platforms_L60D,
       weeks_accessed_L60D,
       unq_recs_L60D,
       avg_hour_L60D,
       plays_L7D,
       t20_plays_L7D,
       recs_L7D,
       day_bounce_rate_L7D,
       brands_played_L7D,
       actions_L7D,
       days_accessed_L7D,
       plays_delta,
       t20_plays_delta,
       recs_delta,
       actions_delta,
       day_bounce_rate_delta,
       brands_played_delta,
       subcats_played_delta,
       CASE WHEN DATE_DIFF('day', access_date, LEAD(access_date) OVER (PARTITION BY user_primaryid ORDER BY access_date ASC)) IS NULL OR DATE_DIFF('day', access_date, LEAD(access_date) OVER (PARTITION BY user_primaryid ORDER BY access_date ASC)) >= 60 THEN 1 ELSE 0 END AS churn_status -- Churn status identifies if access point was one of churn or not
FROM
(SELECT user_primaryid,  
        access_date,
        date_first_login,
        user_type,
        month_access_date,
        first_brand_ranking_indicator,
        plays, 
        CASE WHEN IS_NAN(t20_plays) THEN 0 ELSE t20_plays END AS t20_plays,
        actions,
       -- Activity Last 60 Days
       plays_L60D, 
       CASE WHEN IS_NAN(t20_plays_L60D) THEN 0 ELSE t20_plays_L60D END AS t20_plays_L60D,
       recs_L60D,
       actions_L60D,
       CASE WHEN IS_NAN(day_bounce_rate_L60D) THEN 0 ELSE day_bounce_rate_L60D END AS day_bounce_rate_L60D, 
       brands_played_L60D, 
       subcats_played_L60D, 
       platforms_L60D,
       weeks_accessed_L60D,
       unq_recs_L60D,
       avg_hour_L60D,
       -- Activity Last 7 Days
       plays_L7D,
       CASE WHEN IS_NAN(t20_plays_L7D) THEN 0 ELSE t20_plays_L7D END AS t20_plays_L7D,
       recs_L7D,
       CASE WHEN IS_NAN(day_bounce_rate_L7D) THEN 0 ELSE day_bounce_rate_L7D END AS day_bounce_rate_L7D, 
       brands_played_L7D,
       actions_L7D,
       days_accessed_L7D,
       -- Activity Change Last 7 Days
       CASE WHEN IS_NAN(ROUND(CAST(plays_L7D AS double) / CAST(plays_L14D AS double), 3)) THEN 0 ELSE ROUND(CAST(plays_L7D AS double) / CAST(plays_L14D AS double) - 0.5, 3) END plays_delta,
       CASE WHEN IS_NAN(ROUND(CAST(t20_plays_L7D AS double) / CAST(t20_plays_L14D AS double), 3)) THEN 0 ELSE ROUND(CAST(t20_plays_L7D AS double) / CAST(t20_plays_L14D AS double) - 0.5, 3) END t20_plays_delta,
       CASE WHEN IS_NAN(ROUND(CAST(recs_L7D AS double) / CAST(recs_L14D AS double), 3)) THEN 0 ELSE ROUND(CAST(recs_L7D AS double) / CAST(recs_L14D AS double) - 0.5, 3) END recs_delta,
       CASE WHEN IS_NAN(ROUND(CAST(actions_L7D AS double) / CAST(actions_L14D AS double), 3)) THEN 0 ELSE ROUND(CAST(actions_L7D AS double) / CAST(actions_L14D AS double) - 0.5, 3) END actions_delta,
       CASE WHEN IS_NAN(ROUND(CAST(day_bounce_rate_L7D AS double) / CAST(day_bounce_rate_L14D AS double), 3)) THEN 0 ELSE ROUND(CAST(day_bounce_rate_L7D AS double) - CAST(day_bounce_rate_L14D AS double), 3) END day_bounce_rate_delta,
       CASE WHEN IS_NAN(ROUND(CAST(brands_played_L7D AS double) / CAST(brands_played_L14D AS double), 3)) THEN 0 ELSE ROUND(CAST(brands_played_L7D AS double) / CAST(brands_played_L14D AS double) - 0.5, 3) END brands_played_delta,
       CASE WHEN IS_NAN(ROUND(CAST(subcats_played_L7D AS double) / CAST(subcats_played_L14D AS double), 3)) THEN 0 ELSE ROUND(CAST(subcats_played_L7D AS double) / CAST(subcats_played_L14D AS double) - 0.5, 3) END subcats_played_delta
FROM
(SELECT a.user_primaryid,  
        access_date,
        date_first_login,
        CASE WHEN access_date >= DATE_ADD('day', -60, current_date) THEN 'predict' ELSE 'train' END AS user_type, -- This is used to identify if the user is prediction or training
        MONTH(access_date) month_access_date,
        first_brand_ranking_indicator,
        
        -- Activity on access date
        COUNT(CASE WHEN action = 'play' AND CAST(b.datepartition AS date) = access_date THEN 1 END) plays, 
        ROUND(CAST(COUNT(CASE WHEN action = 'play' AND CAST(b.datepartition AS date) = access_date AND top20_brands_indicator = 1 THEN 1 END) AS double) / CAST(COUNT(CASE WHEN action = 'play' AND CAST(b.datepartition AS date) = access_date THEN 1 END) AS double), 3) t20_plays,
        COUNT(CASE WHEN CAST(b.datepartition AS date) = access_date THEN 1 END) actions,
        
        -- Activity in last 7 days
        COUNT(CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 7 AND CAST(b.datepartition AS date) != access_date THEN 1 END) plays_L7D, 
        ROUND(CAST(COUNT(CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 7 AND CAST(b.datepartition AS date) != access_date AND top20_brands_indicator = 1 THEN 1 END) AS double) / CAST(COUNT(CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 7 AND CAST(b.datepartition AS date) != access_date THEN 1 END) AS double), 3) t20_plays_L7D, 
        COUNT(CASE WHEN custom_rule_id IS NOT NULL AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 7 AND CAST(b.datepartition AS date) != access_date THEN 1 END) recs_L7D, 
        COUNT(CASE WHEN DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 7 AND CAST(b.datepartition AS date) != access_date THEN 1 END) actions_L7D,
        ROUND(CAST(COUNT(DISTINCT CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 7 AND CAST(b.datepartition AS date) != access_date THEN b.datepartition END) AS double) / CAST(COUNT(DISTINCT CASE WHEN DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 7 AND CAST(b.datepartition AS date) != access_date THEN b.datepartition END) AS double), 3) day_bounce_rate_L7D, 
        COUNT(DISTINCT CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 7 AND CAST(b.datepartition AS date) != access_date THEN brand END) brands_played_L7D, 
        COUNT(DISTINCT CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 7 AND CAST(b.datepartition AS date) != access_date THEN subcat END) subcats_played_L7D, 
        COUNT(DISTINCT CASE WHEN DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 7 AND CAST(b.datepartition AS date) != access_date THEN b.datepartition END) days_accessed_L7D,
        
        -- Activity in last 14 days
        COUNT(CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 14 AND CAST(b.datepartition AS date) != access_date THEN 1 END) plays_L14D, 
        ROUND(CAST(COUNT(CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 14 AND CAST(b.datepartition AS date) != access_date AND top20_brands_indicator = 1 THEN 1 END) AS double) / CAST(COUNT(CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 14 AND CAST(b.datepartition AS date) != access_date THEN 1 END) AS double), 3) t20_plays_L14D, 
        COUNT(CASE WHEN custom_rule_id IS NOT NULL AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 14 AND CAST(b.datepartition AS date) != access_date THEN 1 END) recs_L14D, 
        COUNT(CASE WHEN DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 14 AND CAST(b.datepartition AS date) != access_date THEN 1 END) actions_L14D,
        ROUND(CAST(COUNT(DISTINCT CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 14 AND CAST(b.datepartition AS date) != access_date THEN b.datepartition END) AS double) / CAST(COUNT(DISTINCT CASE WHEN DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 14 AND CAST(b.datepartition AS date) != access_date THEN b.datepartition END) AS double), 3) day_bounce_rate_L14D, 
        COUNT(DISTINCT CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 14 AND CAST(b.datepartition AS date) != access_date THEN brand END) brands_played_L14D, 
        COUNT(DISTINCT CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 14 AND CAST(b.datepartition AS date) != access_date THEN subcat END) subcats_played_L14D,
    
        -- Activity in last 60 days
        COUNT(CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 60 AND CAST(b.datepartition AS date) != access_date THEN 1 END) plays_L60D, 
        ROUND(CAST(COUNT(CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 60 AND CAST(b.datepartition AS date) != access_date AND top20_brands_indicator = 1 THEN 1 END) AS double) / CAST(COUNT(CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 60 AND CAST(b.datepartition AS date) != access_date THEN 1 END) AS double), 3) t20_plays_L60D, 
        COUNT(CASE WHEN custom_rule_id IS NOT NULL AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 60 AND CAST(b.datepartition AS date) != access_date THEN 1 END) recs_L60D, 
        COUNT(DISTINCT CASE WHEN DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 60 AND CAST(b.datepartition AS date) != access_date THEN custom_rule_id END) unq_recs_L60D, 
        COUNT(CASE WHEN DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 60 AND CAST(b.datepartition AS date) != access_date THEN 1 END) actions_L60D,
        COUNT(DISTINCT CASE WHEN DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 60 AND CAST(b.datepartition AS date) != access_date THEN custom_platform END) platforms_L60D, 
        ROUND(CAST(COUNT(DISTINCT CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 60 AND CAST(b.datepartition AS date) != access_date THEN b.datepartition END) AS double) / CAST(COUNT(DISTINCT CASE WHEN DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 60 AND CAST(b.datepartition AS date) != access_date THEN b.datepartition END) AS double), 3) day_bounce_rate_L60D,
        COUNT(DISTINCT CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 60 AND CAST(b.datepartition AS date) != access_date THEN brand END) brands_played_L60D, 
        COUNT(DISTINCT CASE WHEN action = 'play' AND DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 60 AND CAST(b.datepartition AS date) != access_date THEN subcat END) subcats_played_L60D, 
        COUNT(DISTINCT CASE WHEN DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 60 AND CAST(b.datepartition AS date) != access_date THEN DATE_TRUNC('week',CAST(b.datepartition AS date)) END) weeks_accessed_L60D,
        ROUND(AVG(CASE WHEN DATE_DIFF('day', CAST(b.datepartition AS date), access_date) <= 60 AND CAST(b.datepartition AS date) != access_date THEN hour_timestamp END), 1) avg_hour_L60D
FROM user_access a
LEFT JOIN beacons b ON a.user_primaryid = b.user_primaryid AND CAST(b.datepartition AS date) <= a.access_date 
WHERE DATE_DIFF('day', CAST(b.datepartition AS date), a.access_date) <= 60 OR 
      CAST(b.datepartition AS date) IS NULL
GROUP BY 1, 2, 3, 4, 5, 6))),

training_set AS (
SELECT CASE WHEN user_type = 'train' THEN NULL ELSE CAST(user_primaryid AS varchar) END user_primaryid, -- UserID only needed for prediction, in training made null
       churn_status,
       user_type,
       COALESCE(days_last_access, 0) days_last_access,
       month_access_date,
       COALESCE(days_first_access, 0) days_first_access,
       first_brand_ranking_indicator,
       plays, 
       t20_plays,
       actions,
       plays_L60D, 
       t20_plays_L60D,
       recs_L60D,
       actions_L60D,
       brands_played_L60D, 
       subcats_played_L60D, 
       platforms_L60D,
       weeks_accessed_L60D,
       unq_recs_L60D,
       CASE WHEN avg_hour_L60D IS NULL THEN 12 ELSE avg_hour_L60D END AS avg_hour_L60D,
       plays_L7D,
       t20_plays_L7D,
       recs_L7D,
       day_bounce_rate_L7D,
       brands_played_L7D,
       actions_L7D,
       days_accessed_L7D,
       plays_delta,
       t20_plays_delta,
       recs_delta,
       actions_delta,
       day_bounce_rate_delta,
       brands_played_delta,
       subcats_played_delta
FROM users_last_actions a
LEFT JOIN days d ON d.datepartition = a.access_date
WHERE ((access_date < DATE_ADD('day', -60, current_date) AND d.datepartition IS NOT NULL) OR
      (churn_status = 1) OR
      (user_type = 'predict' AND max_access = access_date)))
      
select *
from training_set
