import os
import re
import cv2
import numpy as np
import pytesseract
import streamlit as st
from pdf2image import convert_from_path

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="AI Resume Analyser & Job Recommendation",
    page_icon="📄",
    layout="wide"
)

# =========================================================
# TESSERACT
# =========================================================
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# =========================================================
# ROLE-WISE SKILLS
# IMPORTANT: ROLE_SKILLS IS DEFINED BEFORE SKILLS
# =========================================================
ROLE_SKILLS = {
    "Python Developer": [
        "Python", "OOP", "Data Structures", "Git", "GitHub",
        "Django", "Flask", "REST API", "SQL"
    ],
    "Data Analyst": [
        "Python", "Pandas", "NumPy", "SQL", "Excel",
        "Power BI", "Tableau", "Matplotlib", "Statistics"
    ],
    "Data Scientist": [
        "Python", "Pandas", "NumPy", "Scikit-learn", "SQL",
        "Machine Learning", "Statistics", "Matplotlib", "Seaborn"
    ],
    "Machine Learning Engineer": [
        "Python", "NumPy", "Pandas", "Scikit-learn",
        "TensorFlow", "PyTorch", "Machine Learning",
        "Deep Learning", "Git", "Docker"
    ],
    "Backend Developer": [
        "Python", "Django", "Flask", "REST API", "SQL",
        "MySQL", "PostgreSQL", "Git", "Docker"
    ],
    "Web Developer": [
        "HTML", "CSS", "JavaScript", "Git", "GitHub", "REST API"
    ],
    "Java Developer": [
        "Java", "OOP", "Data Structures", "SQL", "MySQL",
        "Spring Boot", "Git", "GitHub"
    ],
    "Software Developer": [
        "Python", "Java", "C++", "OOP", "Data Structures",
        "Git", "GitHub", "SQL"
    ],
    "QA Automation Tester": [
        "Selenium", "Python", "Java", "API Testing",
        "Automation Testing", "SQL", "Git"
    ]
}

# =========================================================
# SKILLS DATABASE
# =========================================================
SKILLS = sorted(set(
    skill
    for role_skills in ROLE_SKILLS.values()
    for skill in role_skills
) | {
    "Pandas", "NumPy", "Matplotlib", "Seaborn", "Scikit-learn",
    "TensorFlow", "PyTorch", "Machine Learning", "Deep Learning",
    "Data Science", "Data Analysis", "Power BI", "Tableau", "Excel",
    "Jupyter Notebook", "VS Code", "AWS", "Docker", "PostgreSQL",
    "MongoDB", "Spring Boot", "JavaScript", "HTML", "CSS", "C++",
    "Java", "Git", "GitHub", "SQL", "MySQL", "Python", "Django",
    "Flask", "REST API", "OOP", "Data Structures", "Statistics",
    "Selenium", "API Testing", "Automation Testing"
})

# Common resume spellings/abbreviations
SKILL_ALIASES = {
    "powerbi": "Power BI",
    "power bi": "Power BI",
    "machine learning": "Machine Learning",
    "ml": "Machine Learning",
    "deep learning": "Deep Learning",
    "dl": "Deep Learning",
    "scikit learn": "Scikit-learn",
    "sklearn": "Scikit-learn",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "postgres": "PostgreSQL",
    "restful api": "REST API",
    "rest api": "REST API",
    "spring": "Spring Boot",
    "spring boot": "Spring Boot",
    "automation testing": "Automation Testing",
    "api testing": "API Testing",
    "data structures": "Data Structures",
    "oop": "OOP"
}

SKILL_RECOMMENDATIONS = {
    "Python": "Strengthen Python fundamentals, functions, OOP and data structures.",
    "Pandas": "Practice data cleaning, filtering, grouping and data manipulation.",
    "NumPy": "Practice arrays, vectorized operations and numerical computing.",
    "SQL": "Practice SELECT, JOIN, GROUP BY, subqueries and window functions.",
    "Statistics": "Learn descriptive statistics, probability, hypothesis testing and regression.",
    "Matplotlib": "Practice Python data visualization with real datasets.",
    "Seaborn": "Practice statistical visualization and dashboard-ready charts.",
    "Machine Learning": "Learn supervised and unsupervised machine learning algorithms.",
    "Scikit-learn": "Build small classification, regression and clustering projects.",
    "Django": "Build a simple CRUD web application using Django.",
    "Flask": "Create REST APIs and small backend applications with Flask.",
    "REST API": "Practice creating and consuming REST APIs.",
    "Git": "Practice branching, commits, merge and Git workflows.",
    "GitHub": "Keep projects documented with README files and clean repositories.",
    "Power BI": "Create dashboards using real-world business datasets.",
    "Excel": "Practice formulas, PivotTables, charts and data cleaning.",
    "Tableau": "Build interactive dashboards and storytelling visualizations.",
    "Selenium": "Practice browser automation with real testing scenarios.",
    "API Testing": "Learn HTTP methods, status codes and API test cases.",
    "Automation Testing": "Build basic automated test suites using Selenium or similar tools."
}

# =========================================================
# TEXT EXTRACTION
# =========================================================
def extract_text(pdf_path):
    try:
        pages = convert_from_path(pdf_path, dpi=250)
    except Exception as e:
        raise RuntimeError(
            "PDF reading failed. Make sure Poppler is installed and available."
        ) from e

    text = ""

    for page in pages:
        image = cv2.cvtColor(
            np.array(page),
            cv2.COLOR_RGB2BGR
        )

        image = cv2.resize(
            image,
            None,
            fx=1.5,
            fy=1.5,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        processed = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        page_text = pytesseract.image_to_string(
            processed,
            config="--oem 3 --psm 3"
        )

        text += page_text + "\n"

    return text


def clean_text(text):
    return " ".join(text.replace("\n", " ").split())


# =========================================================
# SKILL EXTRACTION
# =========================================================
def extract_skills(text):
    text_lower = text.lower()

    corrections = {
        "pytho": "python",
        "nur": "numpy",
        "jupiter notebook": "jupyter notebook",
        "vs coc": "vs code",
        "vs co": "vs code"
    }

    for wrong, correct in corrections.items():
        text_lower = text_lower.replace(wrong, correct)

    found = []

    for skill in SKILLS:
        skill_lower = skill.lower()

        # Avoid false matches such as "c" inside other words.
        if len(skill_lower) <= 2:
            pattern = r"(?<!\w)" + re.escape(skill_lower) + r"(?!\w)"
        else:
            pattern = r"(?<!\w)" + re.escape(skill_lower) + r"(?!\w)"

        if re.search(pattern, text_lower):
            found.append(skill)

    # Add common aliases
    for alias, canonical in SKILL_ALIASES.items():
        if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text_lower):
            if canonical not in found:
                found.append(canonical)

    return sorted(set(found))


# =========================================================
# ROLE MATCHING
# =========================================================
def calculate_role_matches(skills):
    skill_set = {s.lower() for s in skills}
    results = []

    for role, required in ROLE_SKILLS.items():
        matched = [
            skill for skill in required
            if skill.lower() in skill_set
        ]

        fit = (len(matched) / len(required)) * 100

        results.append({
            "role": role,
            "fit": fit,
            "matched": matched,
            "missing": [
                skill for skill in required
                if skill.lower() not in skill_set
            ]
        })

    return sorted(results, key=lambda x: x["fit"], reverse=True)


# =========================================================
# RESUME QUALITY SCORE
# =========================================================
def section_score(text):
    lower = text.lower()

    sections = {
        "skills": ["skills", "technical skills"],
        "education": ["education", "academic"],
        "experience": ["experience", "work experience", "internship"],
        "projects": ["projects", "project"],
        "contact": ["email", "phone", "linkedin", "github"]
    }

    found = 0
    for keywords in sections.values():
        if any(keyword in lower for keyword in keywords):
            found += 1

    return (found / len(sections)) * 100


def calculate_resume_score(text, skills, best_fit):
    role_component = best_fit
    breadth_component = min(len(skills) / 12, 1) * 100
    section_component = section_score(text)

    score = (
        role_component * 0.60
        + breadth_component * 0.25
        + section_component * 0.15
    )

    return round(min(score, 100), 2)


# =========================================================
# POSITIVE ANALYSIS
# =========================================================
def get_strengths(skills):
    strengths = []

    if "Python" in skills:
        strengths.append("You have a useful Python foundation for multiple technology roles.")
    if "SQL" in skills or "MySQL" in skills or "PostgreSQL" in skills:
        strengths.append("Your database/SQL knowledge can support development and data roles.")
    if "Power BI" in skills or "Tableau" in skills:
        strengths.append("Your visualization skills are valuable for analytics-oriented jobs.")
    if "Git" in skills or "GitHub" in skills:
        strengths.append("Version-control knowledge is a positive sign for software teams.")
    if "Machine Learning" in skills or "Scikit-learn" in skills:
        strengths.append("Your machine-learning exposure opens additional technical career paths.")
    if "HTML" in skills and "CSS" in skills:
        strengths.append("Your web fundamentals can support web-development opportunities.")

    if not strengths:
        strengths.append(
            "Your resume already gives a starting point; adding more clearly named skills and projects can strengthen it."
        )

    return strengths


def get_job_advice(role, fit):
    if fit >= 70:
        return (
            f"{role} is a strong match for the skills detected in your resume. "
            "You can confidently explore entry-level or internship opportunities in this area."
        )
    if fit >= 45:
        return (
            f"{role} is a promising option. Your current skills provide a foundation, "
            "and learning a few role-specific skills can improve your opportunities."
        )
    return (
        f"{role} is worth exploring if you are interested in this career path. "
        "A small focused learning plan can increase your role fit."
    )


def get_resume_tips(text):
    lower = text.lower()
    tips = []

    if "project" not in lower:
        tips.append("Add 1–2 strong projects with your contribution, tools used and outcome.")
    if "experience" not in lower and "internship" not in lower:
        tips.append("If you have internship, freelance or practical experience, include it clearly.")
    if "github" not in lower:
        tips.append("Add a GitHub profile with your best projects.")
    if "linkedin" not in lower:
        tips.append("Add a complete LinkedIn profile URL.")
    tips.append("Use measurable results wherever possible, such as accuracy, time saved or records processed.")

    return tips


# =========================================================
# UI
# =========================================================
st.markdown(
    """
    <style>
    .hero {
        padding: 34px 38px;
        border-radius: 24px;
        background: linear-gradient(135deg, #eef2ff 0%, #e0f2fe 50%, #f0fdf4 100%);
        border: 1px solid #dbeafe;
        margin-bottom: 22px;
        box-shadow: 0 8px 28px rgba(30, 64, 175, 0.08);
    }
    .main-title {
        font-size: 42px;
        font-weight: 850;
        line-height: 1.15;
        margin: 0;
        background: linear-gradient(90deg, #4338ca, #0284c7, #059669);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        font-size: 19px;
        color: #334155;
        margin-top: 10px;
        margin-bottom: 18px;
    }
    .tag-row {
        display: flex;
        flex-wrap: wrap;
        gap: 9px;
    }
    .tag {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid #cbd5e1;
        color: #1e293b;
        font-size: 14px;
        font-weight: 650;
    }
    .section-title {
        font-size: 23px;
        font-weight: 800;
        padding: 9px 14px;
        border-left: 5px solid #4f46e5;
        background: linear-gradient(90deg, #eef2ff, #ffffff);
        border-radius: 8px;
        margin-top: 25px;
        margin-bottom: 12px;
    }
    .upload-box {
        padding: 18px;
        border-radius: 16px;
        background: #f8fafc;
        border: 1px dashed #94a3b8;
        margin-bottom: 10px;
    }
    .card {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        margin-bottom: 15px;
        background: #ffffff;
    }
    .score {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
        <div class="main-title">📄 AI Resume Analyser &amp; Job Recommendation</div>
        <div class="subtitle">Analyse your resume, discover your skills, find suitable job roles, and get positive career guidance.</div>
        <div class="tag-row">
            <span class="tag">📊 Resume Score</span>
            <span class="tag">🛠️ Skill Detection</span>
            <span class="tag">🎯 Job Recommendations</span>
            <span class="tag">🚀 Career Guidance</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="upload-box"><b>📤 Upload your resume below</b><br>Get a personalised analysis and discover career opportunities that match your current skills.</div>',
    unsafe_allow_html=True
)
with st.sidebar:
    st.header("💡 How it works")
    st.write("1. Upload your resume")
    st.write("2. Resume text is extracted")
    st.write("3. Skills are detected")
    st.write("4. Suitable job roles are matched")
    st.write("5. You get a positive career plan")
    st.divider()
    st.info(
        "This score is guidance, not a pass/fail judgment. "
        "Every resume can be improved."
    )

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

if uploaded_file is not None:
    if st.button("🔍 Analyze Resume", use_container_width=True):

        temp_pdf = "uploaded_resume.pdf"

        try:
            with st.spinner("Analyzing your resume..."):
                with open(temp_pdf, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                resume_text = extract_text(temp_pdf)

                if not resume_text.strip():
                    st.error(
                        "No readable text was detected. Please upload a clearer PDF."
                    )
                    st.stop()

                clean_resume_text = clean_text(resume_text)
                skills = extract_skills(clean_resume_text)

                role_results = calculate_role_matches(skills)
                best = role_results[0]

                resume_score = calculate_resume_score(
                    clean_resume_text,
                    skills,
                    best["fit"]
                )

            st.success("Resume analysis completed successfully! 🎉")

            # -------------------------------------------------
            # OVERVIEW
            # -------------------------------------------------
            st.markdown('<div class="section-title">📊 Your Resume Analysis</div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Resume Score", f"{resume_score:.0f}%")

            with c2:
                st.metric("Best Fit Role", best["role"])

            with c3:
                st.metric("Skills Detected", len(skills))

            st.progress(int(resume_score))

            st.info(
                "🌟 This score shows how well your current resume aligns "
                "with the available career paths. It is not a rejection score."
            )

            # -------------------------------------------------
            # TOP JOB ROLES
            # -------------------------------------------------
            st.markdown('<div class="section-title">🎯 Jobs You Can Consider Applying For</div>', unsafe_allow_html=True)

            for item in role_results[:5]:
                st.write(
                    f"**{item['role']}** — {item['fit']:.0f}% skill match"
                )
                st.progress(int(item["fit"]))

            st.success(get_job_advice(best["role"], best["fit"]))

            # -------------------------------------------------
            # SKILLS
            # -------------------------------------------------
            st.markdown('<div class="section-title">🛠️ Skills Found in Your Resume</div>', unsafe_allow_html=True)

            if skills:
                st.write(" • ".join(f"`{skill}`" for skill in skills))
            else:
                st.warning(
                    "No predefined skills were detected. "
                    "You can still improve the analyzer by adding skills to the database."
                )

            # -------------------------------------------------
            # BEST ROLE MATCH
            # -------------------------------------------------
            st.markdown(f'<div class="section-title">💼 Why {best["role"]}?</div>', unsafe_allow_html=True)

            if best["matched"]:
                st.write("**Skills supporting this role:**")
                st.write(" • ".join(best["matched"]))
            else:
                st.write(
                    "This role is currently the closest available match. "
                    "Adding role-specific skills can improve the fit."
                )

            # -------------------------------------------------
            # GROWTH SKILLS
            # -------------------------------------------------
            st.markdown('<div class="section-title">📚 Skills That Can Increase Your Opportunities</div>', unsafe_allow_html=True)

            if best["missing"]:
                for skill in best["missing"][:6]:
                    advice = SKILL_RECOMMENDATIONS.get(
                        skill,
                        f"Learn and practice {skill} through a small project."
                    )
                    st.write(f"**{skill}:** {advice}")
            else:
                st.success(
                    "Excellent! The main skills defined for this role are already present."
                )

            # -------------------------------------------------
            # STRENGTHS
            # -------------------------------------------------
            st.markdown('<div class="section-title">💪 Your Resume Strengths</div>', unsafe_allow_html=True)

            for strength in get_strengths(skills):
                st.write(f"✅ {strength}")

            # -------------------------------------------------
            # RESUME IMPROVEMENT
            # -------------------------------------------------
            st.markdown('<div class="section-title">🚀 Simple Ways to Make Your Resume Stronger</div>', unsafe_allow_html=True)

            for i, tip in enumerate(get_resume_tips(clean_resume_text), 1):
                st.write(f"{i}. {tip}")

            # -------------------------------------------------
            # CAREER PLAN
            # -------------------------------------------------
            st.markdown('<div class="section-title">🧭 Suggested Next Step</div>', unsafe_allow_html=True)

            st.write(
                f"**Step 1:** Target roles such as **{best['role']}**."
            )
            st.write(
                "**Step 2:** Build one practical project that demonstrates your strongest skills."
            )
            st.write(
                "**Step 3:** Learn the top 2–3 growth skills listed above."
            )
            st.write(
                "**Step 4:** Apply for internships, fresher and entry-level positions while continuing to learn."
            )

            st.success(
                "🌟 Your resume is a starting point, not a limitation. "
                "Keep building skills, projects and confidence — your opportunities can grow."
            )

            # -------------------------------------------------
            # DOWNLOAD REPORT
            # -------------------------------------------------
            report_lines = [
                "AI RESUME ANALYZER - CAREER GUIDANCE REPORT",
                "=" * 55,
                f"Resume Score: {resume_score:.2f}%",
                f"Best Fit Role: {best['role']}",
                f"Best Role Skill Match: {best['fit']:.2f}%",
                "",
                "TOP JOB ROLES",
                "-" * 30,
            ]

            for item in role_results[:5]:
                report_lines.append(
                    f"{item['role']}: {item['fit']:.2f}%"
                )

            report_lines += [
                "",
                "SKILLS DETECTED",
                "-" * 30,
                ", ".join(skills) if skills else "No predefined skills detected.",
                "",
                "MATCHED SKILLS FOR BEST ROLE",
                "-" * 30,
                ", ".join(best["matched"]) if best["matched"] else "None",
                "",
                "GROWTH SKILLS",
                "-" * 30,
                ", ".join(best["missing"]) if best["missing"] else "None",
                "",
                "POSITIVE MESSAGE",
                "-" * 30,
                "Your resume is a starting point, not a limitation.",
            ]

            st.download_button(
                "⬇️ Download Career Report",
                data="\n".join(report_lines),
                file_name="AI_Resume_Career_Report.txt",
                mime="text/plain",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Analysis error: {e}")

        finally:
            if os.path.exists(temp_pdf):
                try:
                    os.remove(temp_pdf)
                except OSError:
                    pass
