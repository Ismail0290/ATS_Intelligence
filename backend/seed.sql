-- ============================================================
-- Seed: Predefined Job Listings
-- ============================================================

-- Create a system recruiter to own seeded jobs
INSERT INTO users (id, email, hashed_password, full_name, role)
VALUES (
    '00000000-0000-0000-0000-000000000001',
    'system@ats-intelligence.com',
    '$2b$12$placeholder_hashed_password',
    'ATS System',
    'recruiter'
) ON CONFLICT (email) DO NOTHING;

INSERT INTO jobs (title, company, description, required_skills, location, employment_type, salary_range, created_by, is_active)
VALUES
(
    'Senior Data Scientist',
    'TechVision AI',
    'We are looking for a Senior Data Scientist to join our growing AI team. You will build and deploy machine learning models at scale, work with large datasets, and collaborate with engineers to productionize solutions. You should have strong skills in Python, ML frameworks, and cloud platforms. Experience with NLP and deep learning is a strong plus.',
    ARRAY['python', 'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit learn', 'pandas', 'numpy', 'aws', 'docker', 'feature engineering', 'sql'],
    'San Francisco, CA (Remote OK)',
    'Full-time',
    '$140,000 – $180,000',
    '00000000-0000-0000-0000-000000000001',
    TRUE
),
(
    'Machine Learning Engineer',
    'NeuralForge Labs',
    'Join our ML platform team to build infrastructure for training, evaluating, and deploying ML models. You will work with MLOps tools, build pipelines with Airflow and Spark, and ensure model reliability in production. Strong Python and cloud experience required.',
    ARRAY['python', 'machine learning', 'tensorflow', 'pytorch', 'docker', 'kubernetes', 'airflow', 'spark', 'aws', 'mlops', 'ci cd', 'feature engineering'],
    'New York, NY',
    'Full-time',
    '$130,000 – $165,000',
    '00000000-0000-0000-0000-000000000001',
    TRUE
),
(
    'NLP Research Engineer',
    'LanguageAI Corp',
    'We are seeking an NLP Research Engineer passionate about language understanding. You will fine-tune large language models, work with HuggingFace Transformers, and build evaluation pipelines for text classification, NER, and summarization tasks.',
    ARRAY['python', 'nlp', 'deep learning', 'huggingface', 'pytorch', 'tensorflow', 'machine learning', 'scikit learn', 'numpy', 'pandas', 'sql'],
    'Remote',
    'Full-time',
    '$120,000 – $155,000',
    '00000000-0000-0000-0000-000000000001',
    TRUE
),
(
    'Data Engineer',
    'DataStream Inc',
    'Build and maintain scalable data pipelines, ETL workflows, and data warehouses. You will work with Spark, Kafka, Airflow, and dbt to power analytics and ML teams. Strong SQL and Python skills are essential.',
    ARRAY['python', 'sql', 'spark', 'kafka', 'airflow', 'dbt', 'data engineering', 'etl', 'aws', 'gcp', 'docker', 'data analysis'],
    'Austin, TX',
    'Full-time',
    '$110,000 – $145,000',
    '00000000-0000-0000-0000-000000000001',
    TRUE
),
(
    'Full Stack Engineer (AI Products)',
    'Quantum Apps',
    'Build AI-powered web applications end-to-end. You will work on React frontends, Python/Node backends, and integrate ML model APIs. Experience with TypeScript, REST APIs, and cloud deployment required. Bonus for experience with LLM APIs.',
    ARRAY['javascript', 'typescript', 'python', 'api', 'rest', 'docker', 'aws', 'sql', 'system design', 'agile', 'graphql', 'microservices'],
    'Seattle, WA (Hybrid)',
    'Full-time',
    '$115,000 – $150,000',
    '00000000-0000-0000-0000-000000000001',
    TRUE
),
(
    'Cloud Solutions Architect',
    'SkyScale Technologies',
    'Design and implement cloud-native architectures on AWS and GCP. You will create scalable, resilient, and secure cloud infrastructure using Terraform, Kubernetes, and microservices patterns. Strong system design and DevOps experience required.',
    ARRAY['aws', 'gcp', 'azure', 'kubernetes', 'docker', 'terraform', 'microservices', 'system design', 'ci cd', 'python', 'agile', 'cloud'],
    'Chicago, IL',
    'Full-time',
    '$135,000 – $170,000',
    '00000000-0000-0000-0000-000000000001',
    TRUE
),
(
    'Business Intelligence Analyst',
    'InsightEdge Analytics',
    'Turn complex data into actionable insights. Build dashboards, reports, and analytical models using Tableau, Power BI, and SQL. Partner with business teams to define KPIs and drive data-driven decisions.',
    ARRAY['sql', 'tableau', 'power bi', 'data analysis', 'data visualization', 'excel', 'python', 'communication', 'agile'],
    'Boston, MA (Hybrid)',
    'Full-time',
    '$85,000 – $115,000',
    '00000000-0000-0000-0000-000000000001',
    TRUE
),
(
    'AI Product Manager',
    'FutureTech Ventures',
    'Lead the product roadmap for AI-powered SaaS products. Work closely with ML engineers, designers, and business stakeholders to ship impactful features. Strong understanding of ML capabilities, user research, and agile product management required.',
    ARRAY['product management', 'project management', 'agile', 'communication', 'data analysis', 'leadership', 'python', 'machine learning', 'api'],
    'Remote',
    'Full-time',
    '$120,000 – $155,000',
    '00000000-0000-0000-0000-000000000001',
    TRUE
);
